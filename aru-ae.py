# ## Implementing Autoencoder
# 
# https://gist.github.com/AFAgarap/4f8a8d8edf352271fa06d85ba0361f26

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from torch.utils.data import Dataset, DataLoader

import os

import numpy as np


# In[65]:


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

        return torch.tensor(image)


#data = SpectrogramDatasetLoader('/home/hunaid/lucidmonkeys/spectrograms/')
data = SpectrogramDatasetLoader('/net/projects/scratch/winter/valid_until_31_July_2021/0-animal-communication/data_grid/Chimp_IvoryCoast/aru_continuous_recordings/spectrograms/')

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(129 * 129, 100),
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

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model.parameters()

get_ipython().run_line_magic('pinfo', 'torch.nn.Module.parameters')

batch_size = 512
epochs = 20
learning_rate = 1e-1

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu
# model = AE(input_shape=129).to(device)
model = autoencoder().to(device)

# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# mean-squared error loss
criterion = nn.MSELoss()

for epoch in range(epochs):
    loss = 0
    stepsize = 129
    
    for spectrogram in data:
        chops = int(spectrogram.shape[1] / stepsize)
        for _ in range(chops):
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            #batch_features = batch_features.view(-1, 784).to(device)
            snippet = spectrogram[:, 0+(_*stepsize):129+(_*stepsize)]
            snippet = torch.reshape(snippet, (-1,))

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

