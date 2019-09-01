import argparse
import time
import datetime

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
