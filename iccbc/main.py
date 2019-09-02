from iccbc.utils import Config, str2bool
from iccbc.train import train
import argparse

train_config = Config({
    'use_cuda': True,

    # Data
    'dataset_path': '../datasets/yesno',
    'num_train_regular':    58,       # Number of training samples for regular training
    'num_val_regular':      1,        # Number of validation samples for regular training
    'do_overfitting': False,             # Set overfit or regular training
    'num_train_overfit':    2,          # Number of training samples for overfitting test runs
    'num_workers': 4,                   # Number of workers for data loading

    # Training continuation
    'continue_training':   False,      # Specify whether to continue training with an existing model and solver
    'model_path': '../saves/train20190902111643/model50',
    'solver_path': '../saves/train20190902111643/solver50',

    # Hyper parameters
    'max_train_time_s': None,
    'num_epochs': 50,                  # Number of epochs to train
    'batch_size': 1,
    'learning_rate': 1e-3,
    'betas': (0.9, 0.999),              # Beta coefficients for ADAM
    'padding': 512,                     # Needs to be a power of 2

    # Logging
    'log_interval': 1,           # Number of mini-batches after which to print training loss
    'save_interval': 2,         # Number of epochs after which to save model and solver
    'save_path': '../saves',
    'tensorboard_log_dir': '../../tensorboard_log/',

})

eval_config = Config({

})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', default=False, type=str2bool, help='Train model')
    parser.add_argument('--eval', default=False, type=str2bool, help='Evaluate model')

    args = parser.parse_args()

    if args.train:
        train(train_config)

    # if args.eval:
    #     evaluate(eval_config)
