#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 22:09:14 2019

@author: kai
"""

import torch
import torchaudio as ta
from matplotlib import pyplot as plt
import math

"""
no mu transform
pre-emphasis (to amplify high-frequency signals)
segmenting continuous sound into overlapped frames (100ms, 50ms overlap)
apply Hamming window on these frames
Short-Time Fourier Transform
Compute power spectrum
Apply 20 logarithmic mel-scaled filters
"""

def pre_emphasize(waveform, alpha = 0.97):
    emphasized = torch.cat((torch.unsqueeze(waveform[0,0], 0), waveform[0,1:] - alpha * waveform[0, :-1]))
    emphasized = emphasized.unsqueeze(0)
    return emphasized

def generate_mfccs(MFCC_transform, waveform, sample_rate, plot_mfcc = False):
    """Computes MFCCs for waveform
    
    Arguments
    ---------
    MFCC_transform : ta.transforms.MFCC
    waveform
    sample_rate
    plot_mfcc : bool
    
    Returns
    -------
    mfcc : torch.tensor [n channel, n_filt, n windows], time is now window number
    """
    mfcc = MFCC_transform(waveform)
    
    if plot_mfcc:
        plt.figure()
        #plt.imshow(mfcc.log2()[0,:,:].numpy(), cmap='gray')
        plt.imshow(mfcc.log2()[0,:,:].detach().numpy(), cmap='gray')
        plt.show()
        plt.close()
        
    return mfcc

def auditory_frontend(waveform, sample_rate, frontend_config):
    """Perform auditory frontend operations on waveform"""
    
    frame_stride = 1/sample_rate
    if not frontend_config.use_samplerate_for_stride:
        frame_stride = math.floor(frontend_config.frame_stride / frame_stride)*frame_stride + frontend_config.frame_stride #smallest multiple of sample_rate larger than frame stride (keep exact multiple.)
    
    if frontend_config.pre_emphasis:
        waveform = pre_emphasize(waveform, frontend_config.preemphasis_coefficient)
    
    melkwargs = dict({
            #'wkwargs': wkwargs,
            'win_length': int(frontend_config.frame_length * sample_rate),
            'hop_length': int(frame_stride * sample_rate),
            'n_fft': frontend_config.n_fft,
            #'n_filt': 40
            })
    
    MFCC_transform = ta.transforms.MFCC(sample_rate = sample_rate, n_mfcc = frontend_config.n_filt, melkwargs=melkwargs)    
    mfcc = generate_mfccs(MFCC_transform, waveform, sample_rate, frontend_config.plot_mfcc)
    
    if frontend_config.standardize_mfccs:
        for i in range(mfcc.shape[1]):
            mean = mfcc[0,i,:].mean()
            std = mfcc[0,i,:].std()
            mfcc[0,i,:] = (mfcc[0,i,:] - mean) / std
    #print([mfcc[0,i,:].mean() for i in range(mfcc.shape[1])])
        
    return mfcc