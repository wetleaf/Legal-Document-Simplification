'''
Evaluate the SARI score and Other metric
'''

from pathlib import Path
import sys
# sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --

from easse.cli import evaluate_system_output
from contextlib import contextmanager
import json
from preprocessor import Preprocessor
import torch
from preprocessor import get_data_filepath, EXP_DIR, MILDSUM,  REPO_DIR, WIKI_DOC, D_WIKI
from preprocessor import write_lines, yield_lines, count_line, read_lines, generate_hash
from easse.sari import corpus_sari
import time
from utils.D_SARI import D_SARIsent
# from googletrans import Translator
from bart import SumSim
from argparse import ArgumentParser
import nltk
nltk.download('punkt_tab')


@contextmanager
def log_stdout(filepath, mute_stdout=False):
    '''Context manager to write both to stdout and to a file'''

    class MultipleStreamsWriter:
        def __init__(self, streams):
            self.streams = streams

        def write(self, message):
            for stream in self.streams:
                stream.write(message)

        def flush(self):
            for stream in self.streams:
                stream.flush()

    save_stdout = sys.stdout
    log_file = open(filepath, 'w')
    if mute_stdout:
        sys.stdout = MultipleStreamsWriter([log_file])  # Write to file only
    else:
        sys.stdout = MultipleStreamsWriter([save_stdout, log_file])  # Write to both stdout and file
    try:
        yield
    finally:
        sys.stdout = save_stdout
        log_file.close()

# set random seed universal
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Get Config

args = ArgumentParser()

args.add_argument("-model_dirname", "--model_dirname", default="exp_1733416876640828")
args.add_argument("-checkpoint", "--checkpoint", default="checkpoint-epoch=0.ckpt")
args.add_argument("-summary_length", "--summary_length", default=512)
args.add_argument("-map_reduce","--map_reduce", default=False)
args.add_argument("-device","--device",default="cuda")
args.add_argument("-output_length", "--output_length", default=256)

args,_ = args.parse_known_args()
set_seed(42)

# specify the model_name and checkpoint_name

model_dirname = args.model_dirname
checkpoint_path = args.checkpoint

#### Joint model ####
Model = SumSim.load_from_checkpoint(EXP_DIR /  model_dirname / checkpoint_path).to(args.device)
summarizer = Model.summarizer.to(args.device)
simplifier = Model.simplifier.to(args.device)
summarizer_tokenizer = Model.summarizer_tokenizer
simplifier_tokenizer = Model.simplifier_tokenizer
#### Joint model ####



def get_docs(inputs, token_length):

    sources = []
    for input in inputs:
        source = input.split(' ')
        tokenized_source = []
        for i in range(0,len(source),token_length):
            temp = " ".join(source[i:i+token_length]).strip() 
            if temp:
                tokenized_source.append(temp)

        sources.append(tokenized_source)

    return sources

# add_tokens = torch.tensor([18356, 10]).to(device)

def generate(args, sentence, preprocessor=None):
    '''
    This function is for Joint model to generate/predict
    '''

    if args.map_reduce:
        sentences = get_docs([sentence],args.summary_length)[0]
        summary_ids = torch.Tensor([]).to(args.device)
        for sentence in sentences:
            encoding = summarizer_tokenizer(
                [sentence],
                max_length = args.summary_length,
                truncation = True,
                padding = 'max_length',
                return_tensors = 'pt',
            )
            
            summary_id = summarizer.generate(
                encoding['input_ids'].to(args.device),
                num_beams = 15,
                min_length = 10,
                max_length = 128,
                top_k = 80, top_p = 0.97
            )
            # print(summary_id.shape, summary_ids.shape)
            summary_ids = torch.concat((summary_ids,summary_id[0]))
            del encoding
            torch.cuda.empty_cache()

        summary_ids = summary_ids[:args.summary_length].reshape(1,-1).type(torch.long)
    else:
        encoding = summarizer_tokenizer(
                [sentence],
                max_length = args.summary_length,
                truncation = True,
                padding = 'max_length',
                return_tensors = 'pt',
            )
            
        summary_ids = summarizer.generate(
                encoding['input_ids'].to(args.device),
                num_beams = 15,
                min_length = 50,
                max_length = args.summary_length,
                top_k = 80, top_p = 0.97
            )


    summary_atten_mask = torch.ones(summary_ids.shape).to(args.device)
    summary_atten_mask[summary_ids[:,:] == summarizer_tokenizer.pad_token_id] = 0
    

    beam_outputs = simplifier.generate(
        input_ids = summary_ids,
        attention_mask = summary_atten_mask,
        do_sample = True,
        max_length = args.output_length,
        num_beams = 5, #16
        top_k = 80,  #120
        top_p = 0.95, #0.95
        early_stopping = True,
        num_return_sequences = 1,
    )

    del summary_ids
    del summary_atten_mask

    torch.cuda.empty_cache()
    
    sent = simplifier_tokenizer.decode(beam_outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return sent
    
    
def evaluate(orig_filepath, sys_filepath, ref_filepaths):
    orig_sents = read_lines(orig_filepath)
    # NOTE: change the refs_sents if several references are used
    refs_sents = [read_lines(ref_filepaths)]
    #refs_sents = [read_lines(filepath) for filepath in ref_filepaths]

    return corpus_sari(orig_sents, read_lines(sys_filepath), refs_sents)


def simplify_file(args, complex_filepath, output_filepath, features_kwargs=None, model_dirname=None, post_processing=True):
    '''
    Obtain the simplified sentences (predictions) from the original complex sentences.
    '''

    total_lines = count_line(complex_filepath)
    print(complex_filepath)
    print(complex_filepath.stem)

    output_file = Path(output_filepath).open("a")

    for n_line, complex_sent in enumerate(yield_lines(complex_filepath), start=1):
        ### NOTE: Change it when using Single model or Joint model
        output_sents = generate(args, complex_sent, preprocessor=None)
        

        print(f"{n_line+1}/{total_lines}", " : ", output_sents)
        if output_sents:
            output_file.write(output_sents + "\n")
        else:
            output_file.write("\n")

    output_file.close()
    
    if post_processing: post_process(output_filepath)

def post_process(filepath):
    lines = []
    for line in yield_lines(filepath):
        lines.append(line.replace("''", '"'))
    write_lines(lines, filepath)


def evaluate_on_MILDSUM(args, phase, features_kwargs=None,  model_dirname = None):
    dataset = MILDSUM

    model_dir = EXP_DIR / model_dirname
    output_dir = model_dir / 'outputs'

    output_dir.mkdir(parents = True, exist_ok = True)
    #features_hash = generate_hash(features_kwargs)
    output_score_filepath = output_dir / f'score_{dataset}_{phase}.log.txt'
    complex_filepath =get_data_filepath(dataset, phase, 'complex') #_kw_num3_div0.7
    
    if not output_score_filepath.exists() or count_line(output_score_filepath)==0:
        start_time = time.time()
        complex_filepath =get_data_filepath(dataset, phase, 'complex')
        
        #complex_filepath = get_data_filepath(dataset, phase, 'complex_summary_'+str(ratio))
        pred_filepath = output_dir / f'{complex_filepath.stem}.txt'
        ref_filepaths = get_data_filepath(dataset, phase, 'simple')

        if pred_filepath.exists() and count_line(pred_filepath)==count_line(complex_filepath):
            print("File is already processed.")
        else:
            simplify_file(args, complex_filepath, pred_filepath, features_kwargs, model_dirname)

        print("Evaluate: ", pred_filepath)

        with log_stdout(output_score_filepath):
            scores  = evaluate_system_output(test_set='custom',
                                             sys_sents_path=str(pred_filepath),
                                             orig_sents_path=str(complex_filepath),
                                             refs_sents_paths=str(ref_filepaths))


            print("SARI: {:.2f}\t D-SARI: {:.2f} \t BLEU: {:.2f} \t FKGL: {:.2f} ".format(scores['sari'], scores['D-sari'], scores['bleu'], scores['fkgl']))
            # print("{:.2f} \t {:.2f} \t {:.2f} ".format(scores['SARI'], scores['BLEU'], scores['FKGL']))

            print("Execution time: --- %s seconds ---" % (time.time() - start_time))
            return scores['sari']
    else:
        print("Already exists: ", output_score_filepath)
        print("".join(read_lines(output_score_filepath)))


####### MILDSUM #######

def main():

    evaluate_on_MILDSUM(args, phase='test', features_kwargs=None, model_dirname=model_dirname)

if __name__ == '__main__':
    main()