'''
Main Program: 
> python main.py
'''
# -- fix path --

import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
# -- end fix path --

from preprocessor import MILDSUM_SUMMARY,MILDSUM, WIKI_DOC, EXP_DIR
import time
import json

import argparse

from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from optuna.integration import PyTorchLightningPruningCallback

from bart import SumSim, train


def parse_arguments(dataset):
    p = ArgumentParser()
                  
    p.add_argument('--seed', type=int, default=42, help='randomization seed')
    p.add_argument('-dataset','--dataset', default= dataset)
    p.add_argument("-experiment_dir",'--experiment_dir',default=None)
    p = SumSim.add_model_specific_args(p)
    p = pl.Trainer.add_argparse_args(p)
    args,_ = p.parse_known_args()
    return args


# Create experiment directory
def get_experiment_dir(create_dir=True, dir_name = None):

    if not dir_name :
        dir_name = f'{int(time.time() * 1000000)}'

    path = EXP_DIR / f'exp_{dir_name}'

    if create_dir == True: 
        path.mkdir(parents=True, exist_ok=True)

    return path

def log_params(filepath, kwargs):
    filepath = Path(filepath)
    kwargs_str = dict()
    for key in kwargs:
        kwargs_str[key] = str(kwargs[key])
    json.dump(kwargs_str, filepath.open('w'), indent=4)



def run_training(args):

    args.output_dir = get_experiment_dir(True, args.experiment_dir)
    # logging the args
    log_params(args.output_dir / "params.json", vars(args))
    # args.dataset = dataset
    print("Dataset: ",args.dataset)


    train(args)


if __name__ == '__main__':
    dataset = MILDSUM
    args = parse_arguments(dataset)
    run_training(args)

