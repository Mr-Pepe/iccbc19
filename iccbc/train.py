import torch
from torchaudio.datasets import YESNO
import torchaudio as ta
from iccbc.model import WaveNet
from iccbc.solver import Solver
from iccbc.dataset import CustomDataset
from torch.utils.data import DataLoader, SequentialSampler, SubsetRandomSampler
from iccbc.neuromorphic_encoding import neuromorphic_encoding


def train(config):

    # Seed for reproducible results
    seed = 123
    torch.manual_seed(seed)

    # Configure GPU if available
    if config.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(seed)
        kwargs = {'pin_memory': True}
        print("GPU available. Training on {}.".format(device))
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type('torch.FloatTensor')
        kwargs = {}
        print("No GPU. Training on {}.".format(device))

    total_dilation = sum([2 ** i for i in range(config.n_layers_per_block)]) * config.n_blocks

    print("Loading dataset from {}".format(config.dataset_path))
    """
    dataset = CustomDataset(
        path=config.dataset_path,
        sequence_length=config.sequence_length,
        total_dilation=total_dilation,
        transform=ta.transforms.MuLawEncoding(),
        overwrite=config.overwrite,
        plot=False,
        shift=config.shift
    )
    """
    dataset = neuromorphic_encoding(main_config=config)

    if config.batch_size > len(dataset):
        raise Exception('Batch size bigger than the dataset.')

    if config.do_overfitting:
        print("Overfitting on a subset of {} samples".format(config.num_train_overfit))
        if config.batch_size > config.num_train_overfit:
            raise Exception('Batchsize for overfitting bigger than the number of samples for overfitting.')
        else:
            train_data_sampler = SequentialSampler(range(config.num_train_overfit))
            val_data_sampler = SequentialSampler(range(config.num_train_overfit))

    else:
        print("Training on {} samples".format(config.num_train_regular))
        if config.num_train_regular + config.num_val_regular > len(dataset):
            raise Exception(
                'Trying to use more samples for training and validation than len(dataset), {} > {}.'.format(
                    config.num_train_regular + config.num_val_regular, len(dataset)
                ))
        else:
            train_data_sampler = SubsetRandomSampler(range(config.num_train_regular))
            val_data_sampler = SubsetRandomSampler(range(
                config.num_train_regular,
                config.num_train_regular + config.num_val_regular
            ))

    train_data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sampler=train_data_sampler,
        drop_last=True,
        **kwargs
    )

    val_data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sampler=val_data_sampler,
        drop_last=True,
        **kwargs
    )

    print("Initializing model and solver ...")
    model = WaveNet(
        n_blocks=config.n_blocks,
        n_input_channels=config.n_input_channels,
        n_layers_per_block=config.n_layers_per_block,
        n_dilation_channels=config.n_dilation_channels,
        n_skip_channels=config.n_skip_channels,
        n_residual_channels=config.n_residual_channels,
        n_out_channels=config.n_out_channels
    )

    solver = Solver()

    print("Total receptive field is {} samples".format(total_dilation))

    if config.continue_training:
        print("Continuing training with model: {} and solver: {}".format(
            config.model_path, config.solver_path)
        )

        model.load_state_dict(torch.load(config.model_path))
        model.to(device)

        solver.optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        solver.load(config.solver_path, device=device)
        optimizer = None

    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Perform training
    solver.train(model=model,
                 train_config=config,
                 tensorboard_path=config.tensorboard_log_dir,
                 optim=optimizer,
                 num_epochs=config.num_epochs,
                 max_train_time_s=config.max_train_time_s,
                 train_loader=train_data_loader,
                 val_loader=val_data_loader,
                 log_after_iters=config.log_interval,
                 save_after_epochs=config.save_interval,
                 save_path=config.save_path,
                 device=device,
                 do_overfitting=config.do_overfitting,
                 total_dilation=total_dilation)
