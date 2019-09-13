#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 22:27:43 2019

@author: kai

The imported SOM class is taken (with minor modifications) from Giannisnik's https://github.com/giannisnik/som
which is itself an adaptation to PyTorch from the Tensorflow code found in https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow/
Good tutorial on SOMs: http://www.ai-junkie.com/ann/som/som1.html
"""

import torch
import os
import numpy as np
from iccbc.som_som import SOM
from torch.nn.functional import one_hot
import torch.multiprocessing as mp

class SavableSOM(SOM):
    def __init__(self, m, n, dim, niter, alpha=None, sigma=None):
        super(SavableSOM, self).__init__(m, n, dim, niter, alpha, sigma)
    
    def saveweights(self, savepath):
        os.makedirs(savepath, exist_ok=True)
        torch.save(self.weights, os.path.join(savepath, 'som_weights.pt'))
        
    def select_random_mfccs(self, data):
        rand_dataidx = np.random.randint(len(data))
        rand_frame = np.random.randint(data[rand_dataidx].shape[2])
        return data[rand_dataidx][0,:,rand_frame]
        
    def improve_random_start(self, data, pr_its = 1000, n_tries = 5):
        select_it = 0
        qerr = 0
        for iter_no in range(pr_its):
            qerr = qerr + self(self.select_random_mfccs(data), iter_no)[0]
        
        min_qerr = qerr
        best = self.weights
        
        for i_try in range(n_tries):
            self.weights = torch.randn(self.m*self.n, self.dim)
            qerr = 0
            var = 0
            for iter_no in range(pr_its):
                qerrit, varit = self(self.select_random_mfccs(data), iter_no)
                qerr = qerr + qerrit
                var = var + varit
            if qerr < min_qerr:
                best = self.weights
                select_it = i_try
        
        self.weights = best
        if var != 0:
            fvu = qerr/var
        else:
            fvu = 0
        print('Selected weights from try %d. MQE Over %d Iterations: %.3f, FVU: %.3f'\
              %(select_it, pr_its, np.sqrt(min_qerr/pr_its), fvu))
        return best
    
    def train(self, data, pr_its = 1000, n_tries_impr_start = 5, logpath = None):
        """Train the SOM's weights
        
        Arguments
        ---------
        data : list of torch.tensor [n channels, n_filt, number of windows]
        pr_its : int , control frequency of output
        
        """
        
        print('\nTraining SOM...')
        
        qerr = 0
        var = 0
        
        self.improve_random_start(data, pr_its = pr_its, n_tries = n_tries_impr_start)
        
        if logpath is not None:
            os.makedirs(logpath, exist_ok=True)
            f = open(os.path.join(logpath, "somlog.txt"), "w")
            f.write('SOM %s Training' %logpath)
        
        for iter_no in range(self.niter):
            if(iter_no % pr_its == 0):
                if var != 0:
                    fvu = qerr/var
                else:
                    fvu = 0.0
                print("\rIteration %d / %d. MQE Over Last %d Iterations: %.3f, FVU: %.3f"\
                      %(iter_no, self.niter, pr_its, np.sqrt(qerr/pr_its), fvu), end='')
                if logpath is not None:
                    f.write("\nIteration %d / %d. MQE Over Last %d Iterations: %.3f, FVU: %.3f"\
                      %(iter_no, self.niter, pr_its, np.sqrt(qerr/pr_its), fvu))
                qerr = 0
                var = 0
                
            #Select a random frame, pass its MFCC filter bank to the SOM
            qerrit, varit = self(self.select_random_mfccs(data), iter_no)
            qerr = qerr + qerrit
            var = var + varit
        
        if logpath is not None:
            f.close()
                
    def loadweights(self, loadpath):
        self.weights = torch.load(os.path.join(loadpath, 'som_weights.pt'))
        
    def transform_mfcc_seq(self, da, transform_onehot = False):
        """Transforms a single sequence
        
        Arguments
        ---------
        da : torch.tensor [n channels, n_filt, length]
        transform_onehot (optional) : bool, whether to convert output to onehot
        indicating BMU's position or leave as two coordinates
        
        Returns
        -------
        da_tr : torch.tensor [n channels, SOM.m * SOM.n (if onehot), number of windows]
        
        """
        
        da_tr_list = self.map_vects(da.squeeze().transpose(0,1))
        da_tr = torch.stack(da_tr_list).transpose(0,1)
        
        if transform_onehot:
            #convert m, n indices to single integer using standard pairing function,
            #then one_hot the integer
            midx = da_tr.squeeze()[0]
            nidx = da_tr.squeeze()[1]
            paired = nidx*self.m + midx
            #print(midx, nidx, paired)
            da_tr = one_hot(paired.int().long(), self.m*self.n).float().T
            
            """
             oh_M = one_hot(da_tr.squeeze()[0].int().long(), self.m).float().T
             #oh_M = one_hot(da_tr.int().long(), self.m).float().T
             oh_N = one_hot(da_tr.squeeze()[1].int().long(), self.n).float().T
             #oh_N = one_hot(da_tr.int().long(), self.n).float().T
             da_tr = torch.cat((oh_M, oh_N))
             """
             
        da_tr = da_tr.unsqueeze(0)
        
        #print('x', end='')
        
        return da_tr.detach()
        
    def transform(self, data, transform_onehot = False):
        """Transforms a list of tensors with the SOM"""
        
        print('Transforming sequences...\n')
        
        """
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass
        #pool = mp.Pool(processes = mp.cpu_count())
        pool = mp.Pool(processes = 1)
        data_tr = [pool.apply(self.transform_mfcc_seq, args=(da.detach(), transform_onehot)) for da in data]
        data_tr = [p.get() for p in data_tr]
        
        pool.close()
        pool.join()
        """

        data_tr = []
        
        num_tr = 0
        for da in data:
            
            #da_tr.requires_grad_(True)
            data_tr.append(self.transform_mfcc_seq(da, transform_onehot))
            
            num_tr = num_tr + 1
            print('\rTransformed %d sequences.' %num_tr, end='')
        
        return data_tr
