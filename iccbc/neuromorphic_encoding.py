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
from matplotlib import pyplot as plt
import argparse

# For description of configuration options see:
# https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
# and paper on robust sound classification

encoding_config = Config({
        
        #Dataset Paths
        'dataset_path_train': '../datasets/yesno',
        'dataset_path_test': '../datasets/yesno',
        
        #Auditory frontend configuration
        'mu_law_encoding': False,
        'pre_emphasis': False,
        'preemphasis_coefficient': 0.97,
        'frame_length': 0.025, #frame size 25 ms
        'frame_stride': 0.01, #stride 10ms, overlap 15ms
        'n_fft': 512,
        'n_filt': 40,
        'plot_mfcc': False,
        
        #SOM configuration
        'SOM_path': '../saves_SOM/20190907175578/',
        'SOM_m': 8,
        'SOM_n': 8,
        'SOM_niter': 1000000, #one iteration is processing a single time frame, so should be generous
        'SOM_alpha': 0.3,
        'SOM_sigma': 4, #Typical value: max(m, n)/2
            #Previous try: 25, 
        'SOM_print_its': 1000, #Number of iterations after which to update display and average QE
        'transform_onehot': True
        })

def get_frontend_config():
    first = 'mu_law_encoding'
    last = 'plot_mfcc'
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

def neuromorphic_encoding(is_training):
    """This is the encoder's main function. It reads in the dataset and performs 
    both the auditory frontend and SOM based transformations.
    If is_training is True, dataset_path_train is used and the SOM is first 
    trained on the data here before applying the transformation.
    If is_training is False, the SOM is loaded from the SOM_path and the dataset 
    in dataset_path_test immediately transformed.
    Thus, in order to train on the training set and transform it, and then
    transform the test set, one only has to call this function once with
    is_training = True, followed by one with is_training = False.
    
    
    Arguments
    ---------
    is_training : bool, specifies whether to read in and transform the training 
    dataset and train the SOM or read in and transform the test set, loading
    the weights of the SOM from file.
    
    Returns
    -------
    dataset : list of torch.tensor [channels, som.m * som.n, number of windows in seq], encoded
    
    """
    
    dim = encoding_config.n_filt
    som = SavableSOM(encoding_config.SOM_m, encoding_config.SOM_n, dim, encoding_config.SOM_niter, encoding_config.SOM_alpha, encoding_config.SOM_sigma)
    
    if encoding_config.mu_law_encoding:
        mu_law_encoding = ta.transforms.MuLawEncoding()
    else:
        mu_law_encoding = None
       
    if is_training:
        path = encoding_config.dataset_path_train
        dataset = CustomDataset(path, plot=False, transform=mu_law_encoding, frontend_config=get_frontend_config())
        som.train(dataset.data, encoding_config.SOM_print_its)
        som.saveweights(encoding_config.SOM_path)
        print('\nTransforming data...')
        dataset.transform_SOM(som, transform_onehot=encoding_config.transform_onehot)
        dataset.resave()
    else:
        path = encoding_config.dataset_path_test
        som.loadweights(encoding_config.SOM_path)
        dataset = CustomDataset(path, plot=False, transform=mu_law_encoding, frontend_config=get_frontend_config(), som=som)
    
    return dataset

#Test: A = torch.load(os.path.join(path, 'yesno.pt'))
#Test: B = torch.load(os.path.join(path, 'yesno_backup.pt'))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--is_training', default=False, type=str2bool, help='Train SOM')

    args = parser.parse_args()

    neuromorphic_encoding(args.is_training)