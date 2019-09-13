from iccbc.utils import Config, str2bool
from iccbc.train import train
from iccbc.eval import evaluate
import argparse

config = Config({
    'use_cuda': True,

    # Data
    'dataset_path': '../datasets/yesno',
    'overwrite':        False,           # Overwrite existing .pt file
    'num_train_regular':    56,       # Number of training samples for regular training
    'num_val_regular':      2,        # Number of validation samples for regular training
    'do_overfitting': True,             # Set overfit or regular training
    'num_train_overfit':    1,          # Number of training samples for overfitting test runs
    'num_workers': 0,                   # Number of workers for data loading
    'sequence_length': 20000,           # Length of sequences that are sampled from the dataset
    'shift': 1,                         # How far into the future the prediction should be

    # Training continuation
    'continue_training':   False,      # Specify whether to continue training with an existing model and solver
    'model_path': '../saves/train20190902111643/model50',
    'solver_path': '../saves/train20190902111643/solver50',

    # Hyper parameters
    'max_train_time_s': None,
    'num_epochs': 1000,                  # Number of epochs to train
    'batch_size': 1,
    'learning_rate': 1e-3,
    'betas': (0.9, 0.999),              # Beta coefficients for ADAM

    # Model parameters
    'n_input_channels': 400,
    'n_blocks': 2,                      # Number of WaveNet blocks
    'n_layers_per_block': 10,            # Number of dilated layers per block, dilation doubles with every layer
    'n_dilation_channels': 32,          # Number of channels for the gated convolution
    'n_skip_channels': 32,              # Number of channels for the skip connection
    'n_residual_channels': 32,          # Number of channels for the residual path
    'n_out_channels': 32,               # Number of channels for the aggregation


    # Logging
    'log_interval': 1,           # Number of mini-batches after which to print training loss
    'save_interval': 2,         # Number of epochs after which to save model and solver
    'save_path': '../saves',
    'tensorboard_log_dir': '../../tensorboard_log/',

})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', default=False, type=str2bool, help='Train model')
    parser.add_argument('--eval', default=False, type=str2bool, help='Evaluate model')

    args = parser.parse_args()

    if args.train:
        train(config)

    if args.eval:
        evaluate(config)
