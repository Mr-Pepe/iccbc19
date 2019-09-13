#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 22:38:38 2019

@author: kai
"""

import os
import numpy as np
import torchaudio as ta
from iccbc.utils import Config, str2bool
from iccbc.auditory_frontend import auditory_frontend
from iccbc.dataset import CustomDataset
from iccbc.som import SavableSOM
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import argparse

# For description of configuration options see:
# https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
# and paper on robust sound classification

encoding_config = Config({
        
        #Dataset Paths
        'dataset_path': '../datasets/yesno',
        
        #Auditory frontend configuration
        'mu_law_encoding': False,
        'pre_emphasis': False,
        'preemphasis_coefficient': 0.97,
        'frame_length': 0.025, #frame size 25 ms
        'frame_stride': 0.01, #stride 10ms, overlap 15ms
            #eventually, this will be recasted as the smallest multiple of sample_rate larger than frame_stride
        'n_fft': 512,
        'n_filt': 20,
        'plot_mfcc': False,
        'use_samplerate_for_stride': False,
        'standardize_mfccs': True,
        
        #SOM configuration
        'SOM_path': '../saves_SOM/20190912230032/', #fix the way this folder generation is handled
        'SOM_m': 16,
        'SOM_n': 16,
        'SOM_niter': 100000, #one iteration is processing a single time frame, so should be generous
        'SOM_alpha': 1, #initial value 0.3
        'SOM_sigma': 4, #Typical value: max(m, n)/2
            #Previous try: 25, 
        'SOM_print_its': 1000, #Number of iterations after which to update display and average QE
        'SOM_n_tries_improve_start': 5, #Number of random starting weigth combinations to try,
        'SOM_force_retrain': False,
        'transform_onehot': True
        })

def get_frontend_config():
    first = 'mu_law_encoding'
    last = 'standardize_mfccs'
    keys = list(encoding_config.keys())
    return Config(list(encoding_config.items())[keys.index(first):keys.index(last)+1])

def encode(waveform, sample_rate, som = None):
    """dynamically encode a single waveform with the stored SOM weights and MFCC configuration
    
    Arguments
    ---------
    waveform : torch.tensor , no preprocessing
    sample_rate
    som (optional) : SavableSOM , with weights already computed    
    
    Returns
    -------
    transformed : torch.tensor [channels, som.m * som.n (if onehot), number of windows in seq]
    
    """
    
    frontend_config = get_frontend_config()
    frontend_encoded = auditory_frontend(waveform, sample_rate, frontend_config)
    
    if som is None:
        som = SavableSOM(encoding_config.SOM_m, encoding_config.SOM_n, encoding_config.n_filt, encoding_config.SOM_niter, encoding_config.SOM_alpha, encoding_config.SOM_sigma)
        som.loadweights(encoding_config.SOM_path)
        
    transformed = som.transform_mfcc_seq(frontend_encoded, encoding_config.transform_onehot)
    return transformed

def make_dataset(path, main_config, som=None):
    mu_law_encoding = ta.transforms.MuLawEncoding()
    if main_config is None:
        dataset = CustomDataset(path, plot=False, transform=mu_law_encoding, 
                                frontend_config=get_frontend_config(),
                                overwrite=encoding_config.SOM_force_retrain, 
                                som=som)
    else:
        total_dilation = sum([2 ** i for i in range(main_config.n_layers_per_block)]) * main_config.n_blocks
        dataset = CustomDataset(
            path=main_config.dataset_path,
            sequence_length=main_config.sequence_length,
            total_dilation=total_dilation,
            transform=ta.transforms.MuLawEncoding(),
            overwrite=main_config.overwrite,
            plot=False,
            shift=main_config.shift,
            frontend_config = get_frontend_config(),
            som=som
            )
    #if not encoding_config.use_samplerate_for_stride:
    #    dataset.upsample()
    return dataset

def neuromorphic_encoding(train_SOM = None, main_config = None):
    """This is the encoder's main function. It reads in the dataset and performs 
    both the auditory frontend and SOM based transformations.
        
    
    Arguments
    ---------
    train_SOM : bool
    
    Returns
    -------
    dataset : list of torch.tensor [channels, som.m * som.n, number of windows in seq], encoded
    
    """
    
    dim = encoding_config.n_filt
    som = SavableSOM(encoding_config.SOM_m, encoding_config.SOM_n, dim, encoding_config.SOM_niter, encoding_config.SOM_alpha, encoding_config.SOM_sigma)
    
    #if encoding_config.mu_law_encoding:
    #else:
    #   mu_law_encoding = None
    
    if main_config is not None:
        path = main_config.dataset_path
    else:
        path = encoding_config.dataset_path
        
    if train_SOM is None:
        train_SOM = encoding_config.SOM_force_retrain
    
    #print('train SOM: ' , train_SOM)
    
    if not train_SOM and os.path.isfile(os.path.join(encoding_config.SOM_path, 'som_weights.pt')):
        som.loadweights(encoding_config.SOM_path)
        print('Active SOM found.')
        dataset = make_dataset(path, main_config, som)
    else:
        dataset = make_dataset(path, main_config)
        som.train(dataset.data, encoding_config.SOM_print_its, encoding_config.SOM_n_tries_improve_start,
                  encoding_config.SOM_path)
        som.saveweights(encoding_config.SOM_path)
        print('\nTransforming data...')
        dataset.transform_SOM(som, transform_onehot=encoding_config.transform_onehot)
        dataset.resave()     
        
    print('Be sure to manually delete SOM or dataset if the two don\'t correspond.')
    
    return dataset

#Test: A = torch.load(os.path.join(path, 'yesno.pt'))
#Test: B = torch.load(os.path.join(path, 'yesno_backup.pt'))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_SOM', default=False, type=str2bool, help='Train SOM')

    args = parser.parse_args()

    neuromorphic_encoding(args.train_SOM)