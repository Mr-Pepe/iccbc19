import torchaudio as ta
import torch
from iccbc.model import WaveNet
import matplotlib.pyplot as plt
from iccbc.dataset import CustomDataset
from torchaudio.transforms import MuLawDecoding


def evaluate(config):

    SAVE_PATH = '../generated/1.mp3'

    total_dilation = sum([2 ** i for i in range(config.n_layers_per_block)]) * config.n_blocks
    print("Total receptive field is {} samples".format(total_dilation))

    print("Loading dataset from {}".format(config.dataset_path))
    dataset = CustomDataset(
        path=config.dataset_path,
        sequence_length=config.sequence_length,
        total_dilation=total_dilation,
        transform=ta.transforms.MuLawEncoding(),
        overwrite=config.overwrite,
        plot=False
    )

    print("Initializing model ...")
    model = WaveNet(
        n_blocks=config.n_blocks,
        n_input_channels=config.n_input_channels,
        n_layers_per_block=config.n_layers_per_block,
        n_dilation_channels=config.n_dilation_channels,
        n_skip_channels=config.n_skip_channels,
        n_residual_channels=config.n_residual_channels,
        n_out_channels=config.n_out_channels
    )
    model.load_state_dict(torch.load('../saves/train20190903160509/model4'))
    model.eval()

    transformed, _ = dataset[0]

    primer = transformed[:, :3000]

    if torch.cuda.is_available():
        model.to('cuda')
        primer = primer.to('cuda')
    # plt.plot(primer.cpu().t().numpy())
    # plt.show()

    generated = model.generate(primer, 1000)

    plt.plot(generated.cpu().t().numpy())
    plt.show()

    # a, b = ta.load('/home/felipe/Projects/iccbc19/datasets/yesno/0_0_0_1_0_0_0_1.wav')
    ta.save('../generated/3.wav', MuLawDecoding()(generated.cpu()), 8000)

    pass
