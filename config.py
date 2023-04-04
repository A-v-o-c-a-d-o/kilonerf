import torch
import numpy as np
from torch.utils.data import DataLoader


prefix_path = 'path to your project folder'

train_data_path = 'path to training_data.pkl'
test_data_path = 'path to testing_data.pkl'
model_path = 'path to kilonerf.pth'
output_path = 'path to output folder'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
training_dataset = torch.from_numpy(np.load(train_data_path, allow_pickle=True))
train_data_loader = DataLoader(training_dataset, batch_size=2048, shuffle=True)

N = 16
LR = 5e-4
HN=2
HF=6
NBINS=192
EPOCHS=30