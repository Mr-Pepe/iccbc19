from torch.utils.data.dataset import Dataset
from torchaudio.transforms import MuLawEncoding
from torch.nn.functional import one_hot
import torch
import os
import math
import numpy as np
import torchaudio as ta
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from iccbc.utils import detect_leading_silence
from iccbc.auditory_frontend import auditory_frontend


class CustomDataset(Dataset):
    """ This custom datasets reads in all the audio files in a directory and lets
    you sample random sections from the dataset. Supports mp4 and wav for now.  """

    def __init__(self, path, sequence_length=20000, total_dilation=0,
                 overwrite=True, transform=MuLawEncoding(), plot=False, shift=1, frontend_config=None, som=None):
        """ The whole dataset is saved as a Pytorch tensor inside the directory with the name of the directory. If the
        file already exists the audio files are not read in again, unless the override option is set. Samples that are
        shorter than the given sequence length are omitted. 
        The SOM-encoded data is saved in self.data, and the mu-law & one-hot encoded data in self.data_out.
        If frontend_config=som=None, the mu-law encoded data is transferred to self_data and self.data_out set to [].
        """
        
        super(CustomDataset, self).__init__()
        self.data = []
        self.data_out = []
        self.sequence_length = sequence_length
        self.total_dilation = total_dilation
        self.shift = shift
        self.path = path
        
        dir_name = os.path.basename(os.path.normpath(self.path))

        # Check for existence of preprocessed file
        if (frontend_config is not None and som is not None and os.path.isfile(os.path.join(path, '{}.pt'.format(dir_name)))) \
            and os.path.isfile(os.path.join(path, '{}_out.pt'.format(dir_name))) and not overwrite:
            self.data = torch.load(os.path.join(os.path.join(path, '{}.pt'.format(dir_name))))
            self.data_out = torch.load(os.path.join(os.path.join(path, '{}_out.pt'.format(dir_name))))                 
            print('Both datasets loaded.')
        elif frontend_config is None and os.path.isfile(os.path.join(path, '{}.pt'.format(dir_name))) and not overwrite:
            self.data = torch.load(os.path.join(os.path.join(path, '{}.pt'.format(dir_name))))
            print('Mu-law one hot dataset loaded.')

        else:
            #print((frontend_config is None), (som is None), os.path.isfile(os.path.join(path, '{}.pt'.format(dir_name))))
            #print(os.listdir(path))
            num_files = 0

            # Find and add all audio files in directory
            for _, _, fnames in os.walk(path):
                for fname in fnames:
                    if (num_files % 1) == 0:
                        print("\rFound {} sequences.".format(num_files), end='')

                    if fname.lower().endswith(('mp3', 'wav')):
                        
                        waveform, sample_rate = ta.load(os.path.join(path, fname))
                        #print(waveform, waveform.size())

                        if waveform.shape[1] > sequence_length: #had to move check to beginning because after transformation with MFCC it is always too short
                           
                            if plot:
                                plt.plot(waveform.t().numpy())
                                plt.show()
                        
                            dout = transform(waveform).float()
                            
                            # Remove silence from beginning of track
                            #transformed = remove_silence_start_end(transformed)
                            #NEEDS TO BE IMPLEMENTED FOR NON MU DATA
                        
                            dout = one_hot(dout[0, :].int().long(), 256).float().T
                            self.data_out.append(dout)
    
                            if frontend_config is not None:
                                transformed = auditory_frontend(waveform, sample_rate, frontend_config)
                                
                                if som is not None:
                                    transformed = som.transform_mfcc_seq(transformed, transform_onehot=True)
                                    
                                self.data.append(transformed.float())

                        num_files += 1
                        
            if self.data and self.data_out:
                torch.save(self.data, os.path.join(path, '{}.pt'.format(dir_name)))
                torch.save(self.data_out, os.path.join(path, '{}_out.pt'.format(dir_name)))
                #print(self.data[0], self.data[0].shape)
                print('\nBoth datasets saved.')   

            if not self.data and self.data_out:
                self.data = self.data_out
                torch.save(self.data, os.path.join(path, '{}.pt'.format(dir_name)))            
                print('\nMu-law one-hot dataset saved.')
                self.data_out = []

    def __getitem__(self, idx):
        """ Get one sequence from the dataset. idx determines the entry from the dataset. The sequence from that
        entry is sampled randomly. THe output is the same as the input but shifted by 1. """

        sound = self.data[idx]
        
        if self.data_out:
            sound_y = self.data_out[idx]
            sound = sound[0] #Ugly hack bc of mismatch sizes data and data_out, FIX!!!
            
            if sound.shape[1] != sound_y.shape[1]:
                repeat_factor = int(sound_y.shape[1] / sound.shape[1])
                sound = repeat(sound, repeat_factor, 1)
            
        #print(sound.shape)
        #print(idx)
        #print(self.sequence_length)
        #print(self.shift)
        start_idx = torch.randint(high=sound.shape[1] - self.sequence_length - self.shift, size=[1]).item()
        #start_idx = np.random.randint(sound.shape[2] - self.sequence_length - self.shift)

        x = sound[:, start_idx:start_idx+self.sequence_length]
        
        if self.data_out:
            sound = self.data_out[idx]
        
        y = sound[:, start_idx+self.shift:start_idx+self.sequence_length+self.shift]
        y = y[:, self.total_dilation:]
        
        #print(x.size(), y.size())

        return x, y

    def __len__(self):
        return len(self.data)
    
    def transform_SOM(self, som, transform_onehot=True):
        self.data = som.transform(self.data, transform_onehot = transform_onehot)
    
    """
    def upsample(self):
        repeat_factor = int(self.data_out[0].shape[1] / self.data[0].shape[2])
        #print(repeat_factor)
        data_up = []
        for i, da in enumerate(self.data):
            #print(da.squeeze())
            data_up.append(repeat(da.squeeze(), repeat_factor, 1)[:,:self.data_out[i].shape[1]].unsqueeze(0))
            #print(da.shape, self.data[0][0].shape)
        self.data = data_up
    """
    
    def resave(self):
        dir_name = os.path.basename(os.path.normpath(self.path))
        torch.save(self.data, os.path.join(self.path, '{}.pt'.format(dir_name)))
        print('\nProcessed dataset saved.')

def repeat(x, n, dim):
    #https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/9
    if dim == -1:
        dim = len(x.shape) - 1
    #return x.view(int(np.prod(x.shape[:dim+1])), 1, int(np.prod(x.shape[dim+1:]))).repeat(1,n,1).view(*x.shape[:dim], n * x.shape[dim], *x.shape[dim+1:])
    return x.reshape(int(np.prod(x.shape[:dim+1])), 1, int(np.prod(x.shape[dim+1:]))).repeat(1,n,1).reshape(*x.shape[:dim], n * x.shape[dim], *x.shape[dim+1:])
        

def remove_silence_start_end(sound):
    # from https://stackoverflow.com/questions/29547218/remove-silence-at-the-beginning-and-at-the-end-of-wave-files-with-pydub
    start_idx = detect_leading_silence(sound)
    end_idx = detect_leading_silence(sound.flip([1]))

    return sound[0, start_idx:-end_idx].view(1, -1)
