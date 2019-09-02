from torch.utils.data.dataset import Dataset
from torchaudio.transforms import MuLawEncoding
from torch.nn.functional import one_hot
import torch
import os
import torchaudio as ta
import matplotlib.pyplot as plt
from iccbc.utils import moving_average, detect_leading_silence


class CustomDataset(Dataset):
    """ This custom datasets reads in all the audio files in a directory and lets
    you sample random sections from the dataset. Supports mp4 and wav for now.  """

    def __init__(self, path, override=False, transform=MuLawEncoding()):
        """ The whole dataset is saved as a Pytorch tensor inside the directory with the name of the directory. If the
        file already exists the audio files are not read in again, unless the override option is set."""
        super(CustomDataset, self).__init__()
        self.data = []

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
                        # plt.plot(moving_average(transformed.t().numpy(), 10))
                        # plt.show()
                        self.data.append(transformed.float())
                        num_files += 1

            if self.data:
                torch.save(self.data, os.path.join(path, '{}.pt'.format(dir_name)))

    def __getitem__(self, idx):

        x = self.data[idx][0,:-1].view(1, -1)
        y = one_hot(self.data[idx][0, 1:].int().long(), 256).float().view(256, -1)

        return x, y

    def __len__(self):
        return len(self.data)


def remove_silence_start_end(sound):
    # from https://stackoverflow.com/questions/29547218/remove-silence-at-the-beginning-and-at-the-end-of-wave-files-with-pydub
    start_idx = detect_leading_silence(sound)
    end_idx = detect_leading_silence(sound.flip([1]))

    return sound[0, start_idx:-end_idx].view(1, -1)
