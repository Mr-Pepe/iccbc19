from torch.utils.data.dataset import Dataset
from torchaudio.transforms import MuLawEncoding
from torch.nn.functional import one_hot
import torch
import os
import torchaudio as ta


class CustomDataset(Dataset):
    """ This custom datasets reads in all the audio files in a directory and lets
    you sample random sections from the dataset. Supports mp4 and wav for now.  """

    def __init__(self, path, override=False, transform=MuLawEncoding(), padding=512):
        """ The whole dataset is saved as a Pytorch tensor inside the directory with the name of the directory. If the
        file already exists the audio files are not read in again, unless the override option is set."""
        super(CustomDataset, self).__init__()
        self.data = []
        self.padding = padding

        dir_name = os.path.basename(os.path.normpath(path))

        # Check for existence of preprocessed file
        if os.path.isfile(os.path.join(path, '{}.pt'.format(dir_name))) and not override:
            self.data = torch.load(os.path.join(os.path.join(path, '{}.pt'.format(dir_name))))

        else:
            num_files = 0

            # Find and add all audio files in directory
            for _, _, fnames in os.walk(path):
                for fname in fnames:
                    if (num_files % 100) == 0:
                        print("\rFound {} sequences.".format(num_files), end='')

                    if fname.lower().endswith(('mp3', 'wav')):
                        waveform, sample_rate = ta.load(os.path.join(path, fname))
                        transformed = transform(waveform)
                        self.data.append(transformed.float())
                        num_files += 1

            if self.data:
                torch.save(self.data, os.path.join(path, '{}.pt'.format(dir_name)))

        self.data = pad(self.data, n_pad=padding)

    def __getitem__(self, idx):

        x = self.data[idx]
        y = one_hot(self.data[idx].int().long(), 256).float().view(256, -1)

        return x, y

    def __len__(self):
        return len(self.data)


def pad(tensors, n_pad):
    for t in tensors:
        t = torch.cat((torch.zeros((1, n_pad)), t), dim=1)

    return tensors
