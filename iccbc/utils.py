import argparse
import time
import datetime
import numpy as np

class Config(dict):
    """ This Config can be used to access members with dot notation, e.g., config.attribute """

    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        self.__dict__ = self


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def time_left(t_start, n_iters, i_iter):
    iters_left = n_iters - i_iter
    time_per_iter = (time.time() - t_start) / i_iter
    time_left = time_per_iter * iters_left
    time_left = datetime.datetime.fromtimestamp(time_left)
    return time_left.strftime("%H:%M:%S")


def moving_average(a, n=3):
    # From https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def detect_leading_silence(sound, silence_threshold=100, chunk_size=10):
    # from https://stackoverflow.com/questions/29547218/remove-silence-at-the-beginning-and-at-the-end-of-wave-files-with-pydub
    idx = 0 # ms

    assert chunk_size > 0 # to avoid infinite loop
    while (sound[0, idx:idx+chunk_size].max() - sound[0, idx:idx+chunk_size].min()) < silence_threshold and idx < sound.shape[1]-chunk_size:
        idx += chunk_size

    return idx
