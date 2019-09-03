from torch.utils.data.dataset import Dataset
from torchaudio.transforms import MuLawEncoding
from torch.nn.functional import one_hot
import torch
import os
import torchaudio as ta
import matplotlib.pyplot as plt
from iccbc.utils import detect_leading_silence


class CustomDataset(Dataset):
    """ This custom datasets reads in all the audio files in a directory and lets
    you sample random sections from the dataset. Supports mp4 and wav for now.  """

    def __init__(self, path, sequence_length=20000, override=False, transform=MuLawEncoding(), plot=False):
        """ The whole dataset is saved as a Pytorch tensor inside the directory with the name of the directory. If the
        file already exists the audio files are not read in again, unless the override option is set. Samples that are
        shorter than the given sequence length are omitted. """
        super(CustomDataset, self).__init__()
        self.data = []
        self.sequence_length = sequence_length

        dir_name = os.path.basename(os.path.normpath(path))

        # Check for existence of preprocessed file
        if os.path.isfile(os.path.join(path, '{}.pt'.format(dir_name))) and not override:
            self.data = torch.load(os.path.join(os.path.join(path, '{}.pt'.format(dir_name))))

        else:
            num_files = 0

            # Find and add all audio files in directory
            for _, _, fnames in os.walk(path):
                for fname in fnames:
                    if (num_files % 1) == 0:
                        print("\rFound {} sequences.".format(num_files), end='')

                    if fname.lower().endswith(('mp3', 'wav')):
                        waveform, sample_rate = ta.load(os.path.join(path, fname))
                        transformed = transform(waveform)

                        # Remove silence from beginning of track
                        transformed = remove_silence_start_end(transformed)

                        if plot:
                            plt.plot(transformed.t().numpy())
                            plt.show()

                        if transformed.shape[1] > sequence_length:
                            self.data.append(transformed.float())

                        num_files += 1

            if self.data:
                torch.save(self.data, os.path.join(path, '{}.pt'.format(dir_name)))

    def __getitem__(self, idx):
        """ Get one sequence from the dataset. idx determines the entry from the dataset. The sequence from that
        entry is sampled randomly. THe output is the same as the input but shifted by 1. """

        sound = self.data[idx]

        start_idx = torch.randint(sound.shape[1] - self.sequence_length - 1, [1]).item()

        x = sound[0, start_idx:start_idx+self.sequence_length].view(1, -1)
        y = one_hot(sound[0, start_idx+1:start_idx+self.sequence_length+1].int().long(), 256).float().view(256, -1)

        return x, y

    def __len__(self):
        return len(self.data)


def remove_silence_start_end(sound):
    # from https://stackoverflow.com/questions/29547218/remove-silence-at-the-beginning-and-at-the-end-of-wave-files-with-pydub
    start_idx = detect_leading_silence(sound)
    end_idx = detect_leading_silence(sound.flip([1]))

    return sound[0, start_idx:-end_idx].view(1, -1)
