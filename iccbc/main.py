from iccbc.utils import Config, str2bool
from iccbc.train import train
import argparse

DATASET_PATH = '../datasets/yesno'
SAVE_PATH = '../saves'

config = Config({
    'use_cuda': True,

    # Data
    'dataset_path': DATASET_PATH,
    'num_train_regular':    8192,       # Number of training samples for regular training
    'num_val_regular':      512,        # Number of validation samples for regular training
    'do_overfitting': True,            # Set overfit or regular training
    'num_train_overfit':    1,        # Number of training samples for overfitting test runs
    'num_workers': 4,                   # Number of workers for data loading

    # Training continuation
    'continue_training':   False,      # Specify whether to continue training with an existing model and solver
    'model_path': '../saves/',
    'solver_path': '../saves/',

    # Hyper parameters
    'max_train_time_s': None,
    'num_epochs': 600,                  # Number of epochs to train
    'batch_size': 1,
    'learning_rate': 1e-3,
    'betas': (0.9, 0.999),              # Beta coefficients for ADAM
    'padding': 512,                     # Needs to be a power of 2

    # Logging
    'log_interval': 1,           # Number of mini-batches after which to print training loss
    'save_interval': 100,         # Number of epochs after which to save model and solver
    'save_path': SAVE_PATH,
    'log_reconstructed_images': False,  # Show a reconstructed sample after every epoch
    'tensorboard_log_dir': '../../tensorboard_log/',


})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', default=True, type=str2bool, help='Train model')
    parser.add_argument('--eval', default=False, type=str2bool, help='Evaluate model')

    args = parser.parse_args()

    if args.train:
        train(config)
