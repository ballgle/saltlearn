#encoding:utf-8
from os import path
import multiprocessing
from pathlib import Path
"""Note:

"""
BASE_DIR = Path('pybert')

configs = {

    'task':'multi label',
    'data':{
        'raw_data_path': BASE_DIR / 'dataset/raw/train.csv',  # 
        'train_file_path': BASE_DIR / 'dataset/processed/train.tsv',
        'valid_file_path': BASE_DIR / 'dataset/processed/valid.tsv',
        'test_file_path': BASE_DIR / 'dataset/raw/test.csv'
    },
    'output':{
        'log_dir': BASE_DIR / 'output/log', # 
        'writer_dir': BASE_DIR / "output/TSboard",# 
        'figure_dir': BASE_DIR / "output/figure", # 
        'checkpoint_dir': BASE_DIR / "output/checkpoints",# 
        'cache_dir': BASE_DIR / 'model/',
        'result': BASE_DIR / "output/result",
    },
    'pretrained':{
        "bert":{
            'vocab_path': BASE_DIR / 'model/pretrain/uncased_L-12_H-768_A-12/vocab.txt',
            'tf_checkpoint_path': BASE_DIR / 'model/pretrain/uncased_L-12_H-768_A-12/bert_model.ckpt',
            'bert_config_file': BASE_DIR / 'model/pretrain/uncased_L-12_H-768_A-12/bert_config.json',
            'pytorch_model_path': BASE_DIR / 'model/pretrain/pytorch_pretrain/pytorch_model.bin',
            'bert_model_dir': BASE_DIR / 'model/pretrain/pytorch_pretrain',
        },
        'embedding':{}
    },
    'train':{
        'valid_size': 0.2,
        'max_seq_len': 256,
        'do_lower_case':True,
        'batch_size': 4,#24,  # how many samples to process at once
        'epochs': 2,  # number of epochs to train
        'start_epoch': 1,
        'warmup_proportion': 0.1,# Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.
        'gradient_accumulation_steps': 1,# Number of updates steps to accumulate before performing a backward/update pass.
        'learning_rate': 2e-5,
        'n_gpu': [0], # 
        'num_workers': multiprocessing.cpu_count(), # 
        'weight_decay': 1e-5,
        'seed':2018,
        'resume':False,
    },
    'predict':{
        'batch_size':400,
        'amount_of_target_labels': 6
    },
    'callbacks':{
        'lr_patience': 5, # number of epochs with no improvement after which learning rate will be reduced.
        'mode': 'min',    # one of {min, max}
        'monitor': 'valid_loss',  # 
        'early_patience': 20,   # early_stopping
        'save_best_only': True, # SAVE ONLY CHECKPOINT OF EPOCH WITH THE LOWEST LOSS
        'save_checkpoint_freq': 10 # THIS ONLY APPLIES WHEN SAVE_BEST_ONLY = FALSE
    },
    'loss_metrics':{
    #ATTENTION - THIS IS NOT YET IMPLEMENTED - TO CHANGE LOSS FUNCTION, GO TO train_bert_multi_label.py, look under TRAIN_CONFIGS
    #   Loss Functions Currently Available:
    #       Binary_Cross_Entropy - BC



    #    'loss_criterion': 'Binary_Cross_Entropy'
    },


    'label2id' : { # FOR MULTILABEL, ORDERED LABELS GO HERE
        "toxic": 0,
        "severe_toxic": 1,
        "obscene": 2,
        "threat": 3,
        "insult": 4,
        "identity_hate": 5
    },
    'model':{
        'arch':'bert'
    }
}