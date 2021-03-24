#!/net/projects/scratch/winter/valid_until_31_July_2021/hhameed/miniconda3/bin/python3

'''
This file creates a pytorch autoencoder and a dataset loader
It trains the autoencoder on the data
Then it reads the same data to generate bottleneck layer features in a pandas dataframe
Then it performs clustering on the bottleneck features and generates a visualization of the same
Finally it save a csv file with the filename, bottleneck neuron values and the cluster label
'''

from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from numpy import random

from datetime import datetime
import os
import sys
import io

sys.stdout.flush()
# generate folder for all files using current timestamp
folder = datetime.now().strftime("%d%b%Y-%H%M")

os.mkdir(folder)
sys.stdout = open(folder + "/output", "w")
sys.stderr = open(folder + "/error", "w")

# change nbottleneck to the number of neurons in the bottleneck layer
nbottleneck = 7

# define model
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(194 * 257, 24929),
            nn.ReLU(True),
            nn.Linear(24929, 12464),
            nn.ReLU(True),
            nn.Linear(12464, 6232),
            nn.ReLU(True),
            nn.Linear(6232, 3116),
            nn.ReLU(True),
            nn.Linear(3116, 1558),
            nn.ReLU(True),
            nn.Linear(1558, 779),
            nn.ReLU(True),
            nn.Linear(779, 390),
            nn.ReLU(True),
            nn.Linear(390, 195),
            nn.ReLU(True),
            nn.Linear(195, 98),
            nn.ReLU(True),
            nn.Linear(98, 49),
            nn.ReLU(True),
            nn.Linear(49, 25),
            nn.ReLU(True),
            nn.Linear(25, nbottleneck))

        self.decoder = nn.Sequential(
            nn.Linear(nbottleneck, 25),
            nn.ReLU(True),
            nn.Linear(25, 49),
            nn.ReLU(True),
            nn.Linear(49, 98),
            nn.ReLU(True),
            nn.Linear(98, 195),
            nn.ReLU(True),
            nn.Linear(195, 390),
            nn.ReLU(True),
            nn.Linear(390, 779),
            nn.ReLU(True),
            nn.Linear(779, 1558),
            nn.ReLU(True),
            nn.Linear(1558, 3116),
            nn.ReLU(True),
            nn.Linear(3116, 6232),
            nn.ReLU(True),
            nn.Linear(6232, 12464),
            nn.ReLU(True),
            nn.Linear(12464, 24929),
            nn.ReLU(True),
            nn.Linear(24929, 194 * 257),
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def generatefeatures(self, x):
        return self.encoder(x)


# read dataset
class SpectrogramDatasetLoader(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.spectrograms = [f for f in os.listdir(root_dir)]

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.spectrograms[idx]
        img_path = os.path.join(self.root_dir, self.spectrograms[idx])
        image = torch.load(io.BytesIO( open(img_path, "rb").read() ))["data"]

        return { "name" : img_name, "spectrogram" : image }

data = SpectrogramDatasetLoader('/net/projects/scratch/winter/valid_until_31_July_2021/0-animal-communication/data_grid/Chimp_IvoryCoast/detector_train_ds_spec/')

# train model
#batch_size = 50
epochs = 1
learning_rate = 1e-1

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu
model = autoencoder().to(device)

# write model description to standard output
print(model)

# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# mean-squared error loss
criterion = nn.MSELoss()

for epoch in range(epochs):
    loss = 0

    for spectrogram in data:
        # the data array returns a dict of spectrogram and its name
        spectrogram = spectrogram["spectrogram"]

        # if the spectrogram is not of width 194 units, don't run the iteration
        if spectrogram.shape[1] != 194:
            continue
        # reshape mini-batch data to [N, 784] matrix
        # load it to the active device
        #batch_features = batch_features.view(-1, 784).to(device)
        norm = np.linalg.norm(spectrogram)
        snippet = spectrogram / norm
        #snippet = torch.reshape( torch.from_numpy( snippet ), (-1,) ).to(device)
        snippet = torch.reshape( snippet, (-1,) ).to(device)

        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()

        # compute reconstructions
        outputs = model(snippet)

        # compute training reconstruction loss
        train_loss = criterion(outputs, snippet)

        # compute accumulated gradients
        train_loss.backward()

        # perform parameter update based on current gradients
        optimizer.step()

        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()

    # compute the epoch training loss
    loss = loss / len(data)

    # display the epoch training loss
    print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss))
    sys.stdout.flush()

torch.save(model.state_dict(), folder + "/model")

# visualize regenerated samples
os.mkdir("regen-spectrograms")
nsamples = 15

for sample in range(nsamples):
    s = int( np.round( random.uniform(0, len(data)) ) )

    spectrogram = data[s]["spectrogram"]

    # if the spectrogram is not of width 194 units, don't run the iteration
    if spectrogram.shape[1] != 194:
        continue

    norm = np.linalg.norm(spectrogram)
    snippet = spectrogram / norm
    snippet = torch.reshape( snippet, (-1,) ).to(device)

    regen = model(snippet)

    # prepare a white border between to place between original and reproduced
    whiteborder = np.zeros( (50, data[s]["spectrogram"].shape[2]) )

    # reshape snippet into spectrogram shape
    #snippet = snippet.reshape( (data[s].shape[1:2]) )
    snippet = snippet.reshape( (194, 257) )

    # reshape regenerated output into a spectrogram
    regen = regen.detach().numpy().reshape( (194, 257) )

    c = np.concatenate((snippet, whiteborder))
    c = np.concatenate((c, regen))

    plt.matshow(c)
    plt.savefig(folder + "/regen-spectrograms/" + str(s))

# extract features
df = pd.DataFrame( columns = ["filename"] + ["bottleneck" + str(_) for _ in range(nbottleneck)] )
for spectrogram in data:
    imagename = spectrogram["name"]
    spectrogram = spectrogram["spectrogram"]

    # if the spectrogram is not of width 194 units, don't run the iteration
    if spectrogram.shape[1] != 194:
        continue

    norm = np.linalg.norm(spectrogram)
    snippet = spectrogram / norm
    snippet = torch.reshape( snippet, (-1,) ).to(device)

    features = model.generatefeatures(snippet).detach().numpy()
    #features = [str(f.item()) for f in features]
    df = df.append( dict( zip( df.columns, [imagename] + list(features) ) ), ignore_index=True )

df.to_csv(folder + "/features")

# perform clustering
clustering = GaussianMixture(n_components=7, random_state=0).fit(df[[ "bottleneck" + str(_) for _ in range(nbottleneck) ]])

df["clusterlabels"] = clustering.means_

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
r = [-1,1]
X, Y = np.meshgrid(r, r)
ax.scatter3D(points[:, 0], points[:, 1], points[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.savefig(folder + '/gmmmixtures')

# write to file
df.to_csv(folder + "/features")

sys.stdout.close()
sys.stderr.close()
