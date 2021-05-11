#!/net/projects/scratch/winter/valid_until_31_July_2021/hhameed/miniconda3/bin/python3

'''
Train the autoencoder on the data
Then it reads the same data to generate bottleneck layer features in a pandas dataframe
Then it performs clustering on the bottleneck features and generates a visualization of the same
Finally it save a csv file with the filename, bottleneck neuron values and the cluster label
'''


import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np

from numpy import random

from datetime import datetime
import os
import sys
import io

from data.audiodataset import (
    get_audio_files_from_dir,
    Dataset,
)
from models.vanilla_autoencoder import vanillaAutoencoder as autoencoder

parser = argparse.ArgumentParser()

parser.add_argument()

""" Directory parameters """

parser.add_argument(
    "--data_dir", # '/net/projects/scratch/winter/valid_until_31_July_2021/0-animal-communication/data_grid/Chimp_IvoryCoast/xxx'
    type=str,
    help="The path to the dataset directory.",
)

""" Training parameters """

parser.add_argument(
    "--max_train_epochs", type=int, default=500, help="The number of epochs to train for the autoencoder."
)

parser.add_argument(
    "--batch_size", type=int, default=1, help="The number of images per batch."
)

parser.add_argument(
    "--num_workers", type=int, default=4, help="Number of workers used in data-loading"
)

parser.add_argument(
    "--no_cuda",
    dest="cuda",
    action="store_false",
    help="Do not use cuda to train model.",
)

parser.add_argument(
    "--lr",
    "--learning_rate",
    type=float,
    default=1e-5,
    help="Initial learning rate. Will get multiplied by the batch size.",
)

parser.add_argument(
    "--beta1", type=float, default=0.5, help="beta1 for the adam optimizer."
)


""" Input parameters """

parser.add_argument(
    "--sequence_len", type=int, default=1280, help="Sequence length in ms."
)

parser.add_argument(
    "--freq_compression",
    type=str,
    default="linear",
    help="Frequency compression to reduce GPU memory usage. "
    "Options: `'linear'` (default), '`mel`', `'mfcc'`",
)

parser.add_argument(
    "--n_freq_bins",
    type=int,
    default=256,
    help="Number of frequency bins after compression.",
)

parser.add_argument(
    "--n_fft",
    type=int,
    default=4096,
    help="FFT size."
)

parser.add_argument(
    "--hop_length",
    type=int,
    default=441,
    help="FFT hop length."
)

parser.add_argument(
    "--augmentation",
    type=str2bool,
    default=True,
    help="Whether to augment the input data. "
    "Validation and test data will not be augmented.",
)

""" Network parameters """
parser.add_argument(
    "--n_bottleneck", type=int, default=7, help="the number of neurons in the bottleneck layer",
)

ARGS = parser.parse_args()
ARGS.cuda = torch.cuda.is_available() and ARGS.cuda # check if both hardware and user's intention of using cuda
ARGS.device = torch.device("cuda") if ARGS.cuda else torch.device("cpu")

"""
Get audio all audio files from the given data directory except they are broken.
discard the can_load_from_csv() of orcaspot
"""
def get_audio_files():
    audio_files = list(get_audio_files_from_dir(ARGS.data_dir))
    #log.info("Found {} audio files for training.".format(len(audio_files)))
    if len(audio_files) == 0:
        log.close()
        exit(1)
    return audio_files

# create CSV of all training data (exc. noises for augmentation)
#def createCSV(data_dir: str, files: list):
#    csv_file = os.path.join(data_dir, 'ae_training.csv')
#    with open(csv_file, "a") as f:
#        for file in files:
#            f.write(file + "\n")

# read dataset --> this class could be removed
#class SpectrogramDatasetLoader(Dataset):

#    def __init__(self, root_dir, transform=None):
#        """
#        Args:
#            csv_file (string): Path to the csv file with annotations.
#            root_dir (string): Directory with all the images.
#            transform (callable, optional): Optional transform to be applied
#                on a sample.
#        """
#        self.root_dir = root_dir
#        self.spectrograms = [f for f in os.listdir(root_dir)]
#
#   def __len__(self):
#       return len(self.spectrograms)

#   def __getitem__(self, idx):
#       if torch.is_tensor(idx):
#           idx = idx.tolist()

#       img_name = self.spectrograms[idx]
#        img_path = os.path.join(self.root_dir, self.spectrograms[idx])
#        image = torch.load(io.BytesIO( open(img_path, "rb").read() ))["data"]

#       return { "name" : img_name, "spectrogram" : image }


if __name__ == "__main__":

    sys.stdout.flush()
    # generate folder for all files using current timestamp
    folder = datetime.now().strftime("%d%b%Y-%H%M")

    # below 'logging' is tmp, shall be replaced with proper logging
    os.mkdir(folder)
    sys.stdout = open(folder + "/output", "w")
    sys.stderr = open(folder + "/error", "w")

    nbottleneck = ARGS.n_bottleneck

    # for variational autoencoder, could use splits to test how well the learned representation generalize
    # split_fracs = {"train": .7, "val": .15, "test": .15}
    # input_data = DatabaseCsvSplit(
    #    split_fracs, working_dir=ARGS.data_dir, split_per_dir=True
    #)

    audio_files = get_audio_files()

    if ARGS.noise_dir:
        noise_files = [str(p) for p in pathlib.Path(ARGS.noise_dir).glob("*.wav")]
    else:
        noise_files = []

    # Pre-processing is carried out --> input to the network should be (128 x 256)
    dataset = Dataset(
            file_names=audio_files,
            working_dir=ARGS.data_dir,
            cache_dir=ARGS.cache_dir,
            sr=dataOpts["sr"],
            n_fft=dataOpts["n_fft"],
            hop_length=dataOpts["hop_length"],
            n_freq_bins=dataOpts["n_freq_bins"],
            freq_compression=ARGS.freq_compression,
            f_min=dataOpts["fmin"],
            f_max=dataOpts["fmax"],
            seq_len=sequence_len,
            augmentation=ARGS.augmentation,
            noise_files=noise_files,
        )

    dataloaders = torch.utils.data.DataLoader(
            dataset,
            batch_size=ARGS.batch_size,
            shuffle=True,
            num_workers=ARGS.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = autoencoder().to(ARGS.device)

    # write model description to standard output
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=ARGS.lr)

    # mean-squared error loss
    loss_fn = nn.MSELoss()

    # train model
    #batch_size = ARGS.batch_size
    epochs = ARGS.max_train_epochs
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
            train_loss = loss_fn(outputs, snippet)

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
        #if spectrogram.shape[1] != 194:
        #    continue

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

