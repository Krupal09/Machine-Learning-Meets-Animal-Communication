#!/net/projects/scratch/winter/valid_until_31_July_2021/hhameed/miniconda3/bin/python3
# ## Implementing Autoencoder
# 
# https://gist.github.com/AFAgarap/4f8a8d8edf352271fa06d85ba0361f26

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader

import os

import numpy as np

class SpectrogramDatasetLoader(Dataset):

    def __init__(self, root_dir, width, stepsize, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.spectrograms = [f for f in os.listdir(root_dir)]
        self.width = width
        self.stepsize = stepsize

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.spectrograms[idx])
        image = np.load(img_name)

        return image

data = SpectrogramDatasetLoader('/net/projects/scratch/winter/valid_until_31_July_2021/0-animal-communication/data_grid/Chimp_IvoryCoast/aru_continuous_recordings/spectrograms/', 129, 65)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(129 * 129, 100),
            #nn.Linear(33171447, 100),
            nn.ReLU(True),
            nn.Linear(100, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 100),
            nn.ReLU(True), nn.Linear(100, 129 * 129), nn.Tanh())

    def genfeatures(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


batch_size = 512
epochs = 20
learning_rate = 1e-1

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu
# model = AE(input_shape=129).to(device)
model = autoencoder()
model.load_state_dict( torch.load("../../0-animal-communication/autoencoder/1", map_location=device) )
model = model.to(device)

fh = open("features1", "w+")

for epoch in range(epochs):
    loss = 0
    stepsize = 129
    
    for spectrogram in data:
        chops = int(spectrogram.shape[1] / stepsize)
        for _ in range(chops):
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            #batch_features = batch_features.view(-1, 784).to(device)
            norm = np.linalg.norm(spectrogram[:, 0+(_*stepsize):129+(_*stepsize)])
            snippet = spectrogram[:, 0+(_*stepsize):129+(_*stepsize)] / norm
            snippet = torch.reshape( torch.from_numpy( snippet ), (-1,) ).to(device)
            
            features = model.genfeatures(snippet)

            fh.write(str(features[0]) + "," + str(features[1]) + "," + str(features[2]))
