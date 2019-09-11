from torch.utils.data.dataset import Dataset
from torchaudio.transforms import MuLawEncoding
from torch.nn.functional import one_hot
import torch
import os
import torchaudio as ta
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
        shorter than the given sequence length are omitted. """
        super(CustomDataset, self).__init__()
        self.data = []
        self.sequence_length = sequence_length
        self.total_dilation = total_dilation
        self.shift = shift
        self.path = path

        dir_name = os.path.basename(os.path.normpath(self.path))

        # Check for existence of preprocessed file
        if os.path.isfile(os.path.join(path, '{}.pt'.format(dir_name))) and not overwrite:
            self.data = torch.load(os.path.join(os.path.join(path, '{}.pt'.format(dir_name))))
            print('Dataset loaded.')

        else:
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
                        
                        if transform is not None:
                            transformed = transform(waveform).float()
                            # Remove silence from beginning of track
                            transformed = remove_silence_start_end(transformed)
                        else:
                            transformed = waveform

                        if transformed.shape[1] > sequence_length: #had to move check to beginning because after transformation with MFCC it is always too short
                           
                            if plot:
                                plt.plot(transformed.t().numpy())
                                plt.show()
    
                            if frontend_config is not None:
                                transformed = auditory_frontend(transformed, sample_rate, frontend_config)
                                
                                if som is not None:
                                    transformed = som.transform_mfcc_seq(transformed, transform_onehot=True)
                                
                            if frontend_config is None and som is None:
                                transformed = one_hot(transformed[0, :].int().long(), 256).float().T
                            
                            self.data.append(transformed.float())

                        num_files += 1

            if self.data:
                torch.save(self.data, os.path.join(path, '{}.pt'.format(dir_name)))
                print('\nDataset saved.')

    def __getitem__(self, idx):
        """ Get one sequence from the dataset. idx determines the entry from the dataset. The sequence from that
        entry is sampled randomly. The output is the same as the input but shifted by 1. """

        sound = self.data[idx]

        start_idx = torch.randint(sound.shape[1] - self.sequence_length - self.shift, [1]).item()

        x = sound[:, start_idx:start_idx+self.sequence_length]
        y = sound[:, start_idx+self.shift:start_idx+self.sequence_length+self.shift]
        y = y[:, self.total_dilation:]

        return x, y

    def __len__(self):
        return len(self.data)
    
    def transform_SOM(self, som, transform_onehot=True):
        self.data = som.transform(self.data, transform_onehot = transform_onehot)
    
    def resave(self):
        dir_name = os.path.basename(os.path.normpath(self.path))
        torch.save(self.data, os.path.join(self.path, '{}.pt'.format(dir_name)))
        print('\nProcessed dataset saved.')

def remove_silence_start_end(sound):
    # from https://stackoverflow.com/questions/29547218/remove-silence-at-the-beginning-and-at-the-end-of-wave-files-with-pydub
    start_idx = detect_leading_silence(sound)
    end_idx = detect_leading_silence(sound.flip([1]))

    return sound[0, start_idx:-end_idx].view(1, -1)
