import torch
import torch.nn as nn
import abc
import matplotlib.pyplot as plt


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    @abc.abstractmethod
    def forward(self, x):
        pass

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print("Saving model... {}".format(path))
        torch.save(self.cpu().state_dict(), path)


class WaveNet(BaseModel):
    def __init__(self, n_input_channels=256, n_blocks=2, n_layers_per_block=5, n_dilation_channels=32,
                 n_skip_channels=32, n_residual_channels=32, n_out_channels=64):
        super(WaveNet, self).__init__()

        self.n_input_channels = n_input_channels
        self.in_layer = nn.Conv1d(in_channels=n_input_channels, out_channels=n_residual_channels,
                                  kernel_size=1)

        self.blocks = nn.ModuleList()

        for i_block in range(n_blocks):
            self.blocks.append(Block(n_layers_per_block, n_residual_channels, n_skip_channels, n_dilation_channels))

        self.agg_layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(n_skip_channels, n_out_channels, 1),
            nn.ReLU(),
            nn.Conv1d(n_out_channels, 256, 1),
            nn.Softmax(dim=1)
        )
        self.relu = nn.ReLU()

    def forward(self, x):

        residual = self.in_layer(x)
        skip = None

        for i_block in range(len(self.blocks)):
            (residual, skip) = self.blocks[i_block](residual, skip)

        out = self.agg_layers(skip)

        # plt.plot(out[0,:,-1].cpu().detach().numpy())
        # plt.show()

        return out
        # return self.softmax(skip)

    def generate(self, primer, samples=100):

        # Check if in eval mode
        if self.training:
            raise Exception('Set model to eval mode before generating new sound.')

        else:

            sound = primer.clone().detach()
            for i_sample in range(samples):
                pred = self.forward(sound.float().view(1, self.n_input_channels, -1))[0, :, -1]
                sound = torch.cat((sound, pred.float().detach().view(-1, 1)), dim=1)

                if ((i_sample+1) % 100) == 0:
                    print("\rGenerated {}/{} samples.".format(i_sample+1, samples), end='')

        return sound


class Block(nn.Module):
    def __init__(self, n_layers, residual_channels, skip_channels, dilation_channels):
        super(Block, self).__init__()

        self.layers = nn.ModuleList()

        for i_layer in range(n_layers):
            self.layers.append(Layer(residual_channels, skip_channels, dilation_channels, 2**i_layer))

    def forward(self, x, skip):
        for i_layer in range(len(self.layers)):
            (x, skip) = self.layers[i_layer](x, skip)

        return x, skip


class Layer(nn.Module):
    def __init__(self, residual_channels, skip_channels, dilation_channels, dilation):
        super(Layer, self).__init__()

        self.dilation = dilation

        self.gate = nn.Conv1d(residual_channels, dilation_channels, 2, dilation=dilation, stride=1)
        self.filter = nn.Conv1d(residual_channels, dilation_channels, 2, dilation=dilation, stride=1)

        self.skip_squash = nn.Conv1d(dilation_channels, skip_channels, 1)
        self.residual_squash = nn.Conv1d(dilation_channels, residual_channels, 1)

        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

    def forward(self, x, skip_in):

        out = self.tanh(self.filter(x)) * self.sigm(self.gate(x))

        if skip_in is None:
            skip_out = self.skip_squash(out)
        else:
            skip_out = self.skip_squash(out) + skip_in[:, :, self.dilation:]

        residual_out = self.residual_squash(out) + x[:, :, self.dilation:]

        return residual_out, skip_out
