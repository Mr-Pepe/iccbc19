import torchaudio as ta
import torch
from iccbc.model import WaveNet
import matplotlib.pyplot as plt
from iccbc.dataset import CustomDataset
from torchaudio.transforms import MuLawDecoding
# def evaluate(config):

SAVE_PATH = '../generated/1.mp3'


dataset = CustomDataset('../datasets/yesno', transform=ta.transforms.MuLawEncoding(), override=False, plot=False)
transformed, _ = dataset[0]

primer = transformed[0, :8000].view(1, -1)

model = WaveNet()
model.load_state_dict(torch.load('../saves/train20190902214057/model2'))
model.eval()
if torch.cuda.is_available():
    model.to('cuda')
    primer = primer.to('cuda')

# plt.plot(primer.cpu().t().numpy())
# plt.show()

generated = model.generate(primer, 8000)

# plt.plot(generated.cpu().t().numpy())
# plt.show()

# a, b = ta.load('/home/felipe/Projects/iccbc19/datasets/yesno/0_0_0_1_0_0_0_1.wav')
ta.save('../generated/3.wav', MuLawDecoding()(generated.cpu()), 8000)

pass
