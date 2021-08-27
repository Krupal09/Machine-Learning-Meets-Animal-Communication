#!/net/projects/scratch/winter/valid_until_31_July_2021/hhameed/miniconda3/bin/python3

'''
Train the autoencoder on the data
Then it reads the same data to generate bottleneck layer features in a pandas dataframe
Then it performs clustering on the bottleneck features and generates a visualization of the same
Finally it save a csv file with the filename, bottleneck neuron values and the cluster label
'''

import os
import collections
import pathlib
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter
import argparse

from torch.utils.data import Dataset, DataLoader, random_split
from torchsummary import summary # model summary

from data.audiodataset import (
    get_audio_files_from_dir,
    get_broken_audio_files,
    DatabaseCsvSplit,
    DefaultSpecDatasetOps,
    Dataset,
)
from models.plain_autoencoder import Autoencoder as autoencoder
from models.residual_encoder import DefaultEncoderOpts
from models.residual_encoder import ResidualEncoder as Encoder
from models.residual_decoder import DefaultDecoderOpts
from models.residual_decoder import ResidualDecoder as Decoder
from collections import OrderedDict
from utils.logging import Logger
from trainer import Trainer

parser = argparse.ArgumentParser()

"""
Convert string to boolean.
"""
def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

parser.add_argument(
    "-d",
    "--debug",
    dest="debug",
    action="store_true",
    help="Log additional training and model information.",
)

""" Directory parameters """

parser.add_argument(
    "--data_dir", # '/net/projects/scratch/winter/valid_until_31_July_2021/0-animal-communication/data_grid/Chimp_IvoryCoast/xxx'
    type=str,
    help="The path to the dataset directory.",
)

parser.add_argument(
    "--noise_dir",
    type=str,
    default=None,
    help="Path to a directory with noise files used for data augmentation.",
)

parser.add_argument(
    "--model_dir",
    type=str,
    help="The directory where the model will be stored.",
)

parser.add_argument(
    "--cache_dir",
    type=str,
    help="The path to the dataset directory.",
)

parser.add_argument(
    "--checkpoint_dir",
    type=str,
    help="The directory where the checkpoints will be stored.",
)

parser.add_argument(
    "--log_dir", type=str, default=None, help="The directory to store the logs."
)

parser.add_argument(
    "--summary_dir",
    type=str,
    help="The directory to store the tensorboard and delve summaries.",
)


""" Training parameters """

parser.add_argument(
    "--start_from_scratch",
    dest="start_scratch",
    action="store_true",
    help="Start taining from scratch, i.e. do not use checkpoint to restore.",
)

parser.add_argument(
    "--max_train_epochs", type=int, default=500, help="The number of epochs to train for the autoencoder."
)

parser.add_argument(
    "--epochs_per_eval",
    type=int,
    default=2,
    help="The number of batches to run in between evaluations.",
)

# patience: after 'patience' number of epochs (training), terminate if no improvement on validation set
parser.add_argument(
    "--early_stopping_patience_epochs",
    metavar="N",
    type=int,
    default=20,
    help="Early stopping (stop training) after N/epochs_per_eval epochs without any improvements on the validation set.",
)

parser.add_argument(
    "--batch_size", type=int, default=1, help="The number of images per batch."
)

parser.add_argument(
    "--num_workers", type=int, default=4, help="Number of processes to load and process each batch in the background, while the main training loop is busy"
)

parser.add_argument(
    "--no_cuda",
    dest="cuda",
    action="store_false", # default to True when the command-line argument is not present
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

parser.add_argument(
    "--model", type=str, default=None, help="Indicate which model to use. Options are: plain_ae and conv_ae"
)


""" Input parameters """
parser.add_argument(
    "--filter_broken_audio", action="store_true", help="Filter by a minimum loudness using SoX (Sound exchange) toolkit (option could only be used if SoX is installed)."
)

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

parser.add_argument(
    "--conv_kernel_size", nargs="*", type=int, help="Initial convolution kernel size." # e.g. 7
)

parser.add_argument(
    "--max_pool",
    type=int,
    default=None,
    help="Use max pooling after the initial convolution layer. For details, check residual_encoder.py",
)

ARGS = parser.parse_args()
ARGS.cuda = torch.cuda.is_available() and ARGS.cuda # check if both hardware and user's intention of using cuda
ARGS.device = torch.device("cuda") if ARGS.cuda else torch.device("cpu")

if ARGS.conv_kernel_size is not None and len(ARGS.conv_kernel_size):
    ARGS.conv_kernel_size = ARGS.conv_kernel_size[0]

log = Logger("TRAIN", ARGS.debug, ARGS.log_dir)

"""
Get audio all audio files from the given data directory except they are broken.
discard the can_load_from_csv() of orcaspot
"""
def get_audio_files():
    audio_files = list(get_audio_files_from_dir(ARGS.data_dir))
    log.info("Found {} audio files for training.".format(len(audio_files)))
    if len(audio_files) == 0:
        log.close()
        exit(1)
    return audio_files

#def get_audio_files():
#    audio_files = None
#    if input_data.can_load_from_csv():
#        print("Found csv files in {}".format(ARGS.data_dir))
#        #log.info("Found csv files in {}".format(ARGS.data_dir))
#    else:
#        print("Searching for audio files in {}".format(ARGS.data_dir))
#        #log.debug("Searching for audio files in {}".format(ARGS.data_dir))
#       if ARGS.filter_broken_audio:
#            data_dir_ = pathlib.Path(ARGS.data_dir)
#            audio_files = get_audio_files_from_dir(ARGS.data_dir)
#            print("Moving possibly broken audio files to .bkp:")
#            #log.debug("Moving possibly broken audio files to .bkp:")
#            broken_files = get_broken_audio_files(audio_files, ARGS.data_dir)
#            for f in broken_files:
#                print(f)
#                #log.debug(f)
#                bkp_dir = data_dir_.joinpath(f).parent.joinpath(".bkp")
#                bkp_dir.mkdir(exist_ok=True)
#                f = pathlib.Path(f)
#                data_dir_.joinpath(f).rename(bkp_dir.joinpath(f.name))
#        audio_files = list(get_audio_files_from_dir(ARGS.data_dir))
#        print("Found {} audio files for training.".format(len(audio_files)))
#        #log.info("Found {} audio files for training.".format(len(audio_files)))
#        if len(audio_files) == 0:
#            #log.close()
#            exit(1)
#    return audio_files


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


# a dictionary that keeps saving the activations as they come
activations = collections.defaultdict(list)
# get_activation of each layer at each epoch;
# adapted from
# https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/2
def get_activation(layer_name, epoch):
    def hook(module, input, output): # module is the layer that we are interested in
        name = "{}_epoch_{}".format(layer_name.split(".")[1], epoch)
        activations[name].append(output.detach()) # output.cpu()
    return hook

class SaveOutput:
    """
    defining the hook to capture activations of each layer
    adapted from: https://github.com/cosmic-cortex/pytorch-hooks-tutorial/blob/master/hooks.ipynb
    https://gist.github.com/Tushar-N/680633ec18f5cb4b47933da7d10902af
    """

    def __init__(self):
        self.outputs = [] # this can be changed to dict

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    # delete the hook
    def clear(self):
        self.outputs = []

def save_decod_spec(spec, epoch):
    spec = spec.view(spec.size(0), 1, 128, 256)
    save_image(spec, os.path.join(ARGS.decod_dir, 'reconstructed_epoch_{}.png'.format(epoch)))

def module_output_to_numpy(tensor):
    return tensor.detach().to('cpu').numpy()

def save_model(encoder, encoderOpts, decoder, decoderOpts, dataOpts, path):
    """
    Save the trained model and corresponding options.

    To save multiple components, organize them in a dictionary and use torch.save() to serialize the dictionary
    state_dict() for serialization
    """
    encoder = encoder.cpu()
    encoder_state_dict = encoder.state_dict()
    decoder = decoder.cpu()
    decoder_state_dict = decoder.state_dict()


    save_dict = {
        "encoderOpts": encoderOpts,
        "decoderOpts": decoderOpts,
        "dataOpts": dataOpts,
        "encoderState": encoder_state_dict,
        "decoderState": decoder_state_dict,
    }
    if not os.path.isdir(ARGS.model_dir):
        os.makedirs(ARGS.model_dir)
    torch.save(save_dict, path)


if __name__ == "__main__":

    """------------- model-related preparation -------------"""
    # load and update the options for setting up the model
    if ARGS.model is not None:
        if ARGS.model == "plain_ae":
            log.debug("Plain autoencoder is chosen. n_bottleneck: {}".format(ARGS.n_bottleneck))
        elif ARGS.model == "conv_ae":
            encoderOpts = DefaultEncoderOpts
            decoderOpts = DefaultDecoderOpts

            # update the respective parameters if given in terminal
            for arg, value in vars(ARGS).items():
                if arg in encoderOpts and value is not None:
                    encoderOpts[arg] = value
                if arg in decoderOpts and value is not None:
                    decoderOpts[arg] = value
        else:
            raise ValueError("Expected plain_ae or conv_ae as model but received: {}".format(ARGS.model))
    else:
        raise ValueError("--model could not be None. Pls choose one model: plain_ae or conv_ae")

    log.info("Setting up model")
    # create a model and load it to the specified device, either gpu or cpu
    if ARGS.model is not None:
        if ARGS.model == "plain_ae":
            model = autoencoder(ARGS.n_bottleneck).to(ARGS.device)

        elif ARGS.model == "conv_ae":
            encoder = Encoder(encoderOpts).to(ARGS.device)
            decoder = Decoder(decoderOpts).to(ARGS.device)
            model = nn.Sequential(
                OrderedDict([("encoder", encoder), ("decoder", decoder)])
            )
            #log.debug("Encoder: " + str(encoder))
            #log.debug("Decoder: " + str(decoder))

    # write model description to standard output, uses summary from torchsummary
    log.debug(summary(model, (1, 128, 256)))

    """-------------- data-related preparation ------------------"""

    # for variational autoencoder, could use splits to test how well the learned representation generalize
    #split_fracs = {"train": .8, "val": .1, "test": .1}
    #input_data = DatabaseCsvSplit(
    #    split_fracs, working_dir=ARGS.data_dir, split_per_dir=True
    #)

    audio_files = get_audio_files()
    #audio_files = get_audio_files(ARGS.data_dir)

    if ARGS.noise_dir:
        noise_files = [str(p) for p in pathlib.Path(ARGS.noise_dir).glob("*.wav")]
    else:
        noise_files = []

    dataOpts = DefaultSpecDatasetOps
    for arg, value in vars(ARGS).items():
        if arg in dataOpts and value is not None:
            dataOpts[arg] = value
    # log the dataOpts
    log.debug("dataOpts: " + json.dumps(dataOpts, indent=4))

    sequence_len = int(
        float(ARGS.sequence_len) / 1000 * dataOpts["sr"] / dataOpts["hop_length"]
    )
    log.debug("Training with sequence length: {}".format(sequence_len))
    #input_shape = (ARGS.batch_size, 1, dataOpts["n_freq_bins"], sequence_len)  # (256, 128)
    #log.debug("input shape is {}".format(input_shape))


    # Pre-processing is carried out --> input to the network should be (batch_size, 128 x 256)
    # .spec is prepared here
    # 'def load' belongs to class CsvSplit(object)
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

    log.info("Splitting the dataset into train and validation...")
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    log.info("{} audio files in train_ds".format(len(train_ds)))
    log.info("{} audio files in val_ds".format(len(val_ds)))

    #print("batch_size is ", ARGS.batch_size)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=ARGS.batch_size,
            num_workers=ARGS.num_workers,
            pin_memory=True,
        )
    train_dataloader = torch.utils.data.DataLoader(
            train_ds, #dataset,
            batch_size=ARGS.batch_size,
            shuffle=True,
            num_workers=ARGS.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    val_dataloader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=ARGS.batch_size,
        shuffle=True,
        num_workers=ARGS.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    # option: do train-validation-test split to have test_dataset to get final metrics on the best model

    """---------------- training and validation related --------------------------"""

    #epochs = ARGS.max_train_epochs

    """helper tool to visualize the flow of the network and how the shape of the data changes from layer to layer
       usage: tensorboard --logdir=directory_as_this_script runs
       Could be commented out if not wanted
    """
    tb = SummaryWriter()
    # create a single batch of tensor of images
    imgs, _ = next(iter(train_dataloader))
    #for img in imgs:
        #print("max is ", torch.max(img))
        #print("min is ", torch.min(img))
    grid = make_grid(imgs)
    tb.add_image("input examples", grid)
    #tb.add_graph(model, imgs)
    tb.flush()
    tb.close()

    metric_mode = "min"

    trainer = Trainer(
        #model_name=ARGS.model,
        model=model,
        logger=log,
        prefix=ARGS.model,
        checkpoint_dir=ARGS.checkpoint_dir,
        summary_dir=ARGS.summary_dir,
        n_summaries=4,
        start_scratch=ARGS.start_scratch,
    )

    # option: metrics as dict could be defined here
    # metrics = {}

    metric_mode="min"

    optimizer = optim.Adam(model.parameters(), lr=ARGS.lr)

    model = trainer.fit(
        train_dataloader,
        val_dataloader,
        train_ds,
        val_ds,
        loss_fn = nn.MSELoss(), # mean-squared error loss as objective
        optimizer=optimizer,
        n_epochs=ARGS.max_train_epochs,
        val_interval=ARGS.epochs_per_eval,
        patience_early_stopping=ARGS.early_stopping_patience_epochs,
        device=ARGS.device,
        metrics=[],
        val_metric = "loss",
        val_metric_mode = "min",
    )

    path = os.path.join(ARGS.model_dir, "{}_model.pk".format(ARGS.model))

    if ARGS.model == "plain_ae":
       log.error("Save plain_ae model is not supported at the moment.")
    elif ARGS.model == "conv_ae":
        encoder = model.encoder
        decoder = model.decoder
        save_model(
            encoder, encoderOpts, decoder, decoderOpts, dataOpts, path
        )
    else:
        log.error("The model type you would like to save is not supported at the moment. Pls implement.")

    """
    Clustering for convolutional autoencoder :
    * Iterate through the whole dataset(both training and validation) passing them 
      through only encoder to get bottleneck vectors for each data point.
    * Cluster these bottleneck vectors using unsupervised clustering methods like 
      Kmeans or GaussianMixture modeling.
    """
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    # local system hangs while processing GMM so commented out for now
    #from sklearn.mixture import GaussianMixture 

    file_names = []
    bottleneck_outputs = []
    
    with torch.no_grad():
        for i, (input_specs, label) in enumerate(dataloader):
            
            # remove file path to have only file name, ex : ['path/to/directory/file_1.wav']
            file_name = str(label['file_name'])[::-1] # reverse string
            file_name = file_name.split("/")[0]
            file_name = file_name[::-1].split("'")[0] # align it back to the right order
            file_names.append(file_name)     
            #log.info("File-name : {}".format(file_name))
           
            input_specs = input_specs.to(device=ARGS.device)           
            bottleneck_output = model.encoder(input_specs)
            bottleneck_output = np.reshape(bottleneck_output.detach().numpy(), newshape=(-1))
            bottleneck_outputs.append(bottleneck_output)
            
    # add ARGS for number of clusters
    km = KMeans(n_clusters=2, random_state=0)
    #gm = GaussianMixture(n_components=2, random_state=0)

    pred_km = km.fit_predict(bottleneck_outputs)
    #pred_gm = gm.fit_predict(bottleneck_outputs) 

    log.info("predictions : {}".format(pred_km))
    print("Cluster centers of Kmeans : ",km.cluster_centers_)

    #log.info("predictions : {}".format(pred_gm)) 
    #print("Cluster centers of GaussianMixture : {:.8f}".format(gm.means_))
    
    df = pd.DataFrame( columns = ["filename"] + ["cluster_number"] )
    
    # print file names with respective cluster numbers
    for i in range(len(dataloader)):
        log.info("file name : {}, predicted cluster - Kmeans : {}".format(file_names[i], pred_km[i]))
        #log.info("file name : {}, predicted cluster - GaussianMixture : {}".format(file_names[i], pred_gm[i]))        
        
        df = df.append( dict( zip( df.columns, [file_names[i]] + [pred_km[i]] ) ), ignore_index=True )     
    
    summary_dir = ARGS.summary_dir
    if summary_dir is not None:
        df.to_csv(summary_dir + "/Kmeans_clusters")
    
    log.close()

    """Leftover from previous trials. Could be removed when finalizing the script"""

    #save_model(encoder, encoderOpts)

    #train_loss = []
    #val_loss = []
    #start = time.time()

    #for epoch in range(ARGS.max_train_epochs):
        ##train_running_loss = 0.0
        #log.info("Epoch {} is running".format(epoch))

        #if epoch% 5 == 0:
            #hook_handles = []
            #for name, layer in model.named_modules():
                #print("layer is ", layer)
                #if isinstance(layer, torch.nn.ReLU):
                    # register the hook to collect the outputs of the layers of our interest
                    # for plain_ae: nn.ReLU
                    #handle = layer.register_forward_hook(get_activation(name, epoch))
                    #hook_handles.append(handle)

        #for batch_id, (specs,_) in enumerate(dataset):
        #for batch_id, (specs, _) in enumerate(train_dataloaders):
        ##for train_specs, _ in train_dataloader:
            #print(train_specs.shape)
            #print("_ is ", _["ground_truth"].shape)
            ##train_specs = train_specs.to(ARGS.device)

            #if batch_id == 0:
                #tb = SummaryWriter()
                #tb.add_image("epoch_{}_batch_{}_original".format(epoch, batch_id), specs)

            # original code: commented out
            #print("The shape of the specs is ", specs.size())
            # the data array returns a dict of spectrogram and its name
            #spectrogram = spectrogram["spectrogram"]

        # if the spectrogram is not of width 194 units, don't run the iteration
            #if spectrogram.shape[1] != 194:
                #continue
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            #batch_features = batch_features.view(-1, 784).to(device)
            #norm = np.linalg.norm(spectrogram)
            #snippet = spectrogram / norm
            #snippet = torch.reshape( torch.from_numpy( snippet ), (-1,) ).to(device)
            #snippet = torch.reshape( snippet, (-1,) ).to(device)

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            ##optimizer.zero_grad()

            # compute reconstructions
            ##outputs = model(train_specs)
            #if batch_id == 0:
                #tb.add_image("epoch_{}_batch_{}_reconstructed".format(epoch, batch_id), outputs)
                #tb.close()

            # compute training reconstruction loss
            ##loss = loss_fn(outputs, train_specs)

            # compute accumulated gradients
            ##loss.backward()

            # perform parameter update based on current gradients
            ##optimizer.step()

            # add the mini-batch training loss to epoch loss
            # the value of total cost averaged across all training examples of the current batch
            # loss.item()*data.size(0): total loss of the current batch (not averaged).
            ##train_running_loss += loss.item() * train_specs.size(0)

        #if epoch % 5 == 0:
            #for handle in hook_handles:
                #handle.remove()
            #print("Evaluating...")
            #test_imgs, _ = next(iter(test_dataloaders))
            #test_imgs = test_imgs.to(ARGS.device)
            #recon_test = model(test_imgs)
            #tb = SummaryWriter()
            #grid = make_grid(imgs)
            #tb.add_image("Epoch - {}".format(epoch), grid)
            # tb.add_graph(model, imgs)
            #tb.close()
            #print("Finished saving reconstructed...")

        #train_epoch_loss = fit(
        #    model, train_dataloader, train_ds, optimizer, loss_fn
        #)
        #tb.add_scalar("Loss/train", train_epoch_loss, epoch)
        #train_loss.append(train_epoch_loss)

        # display the epoch training loss
        #log.info("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, train_epoch_loss))

        #val_epoch_loss = validate(
        #    model, val_dataloader, val_ds, loss_fn, tb
        #)
        #tb.add_scalar("Loss/validation", val_epoch_loss, epoch)
        #val_loss.append(val_epoch_loss)
        #log.info("epoch : {}/{}, val loss = {:.8f}".format(epoch + 1, epochs, val_epoch_loss))

        #if ARGS.early_stopping:
        #    early_stopping(val_epoch_loss)
        #    if early_stopping.early_stop:
        #        break

        #print(f"Train Loss: {train_epoch_loss:.4f}")
        #print(f"Val Loss: {val_epoch_loss:.4f}")

    #end = time.time()
    #log.info(f"Training time: {(end - start)/60:.3f} minutes")


    #log.close()

        # save output every 5 epoch
        #if epoch % 5 == 0:
            #print(outputs)
            #save_decod_spec(outputs.cpu().data, epoch)

    # concatenate all the outputs we saved to get the the activations for each layer for the whole dataset
    #activations = {name: torch.cat(acts, 0) for layer, acts in activations.items()}

    #plt.figure(figsize=(20,20), frameon=False)
    #for layer_epoch, act in activations.items():
        # just print out the sizes of the saved activations as a sanity check
        # print(layer_epoch, len(act))

        # instead of plotting the activations of the entire dataset at this layer and this epoch
        # plot the last 16 activations
        # fig, axarr = plt.subplots(4,4)
        #for idx in range(16):
            #axarr[idx].imshow(act[-idx].numpy())


    #path = os.path.join(ARGS.model_dir, "plain_ae.pk")
    #torch.save(model.state_dict(), path)

    #images = module_output_to_numpy(save_)


    # visualize regenerated samples
    #os.mkdir("regen-spectrograms")
    #nsamples = 15

    #for sample in range(nsamples):
        #s = int( np.round( random.uniform(0, len(data)) ) )

        #spectrogram = data[s]["spectrogram"]

        # if the spectrogram is not of width 194 units, don't run the iteration
        #if spectrogram.shape[1] != 194:
            #continue

        #norm = np.linalg.norm(spectrogram)
        #snippet = spectrogram / norm
        #snippet = torch.reshape( snippet, (-1,) ).to(device)

        #regen = model(snippet)

        # prepare a white border between to place between original and reproduced
        #whiteborder = np.zeros( (50, data[s]["spectrogram"].shape[2]) )

        # reshape snippet into spectrogram shape
        #snippet = snippet.reshape( (data[s].shape[1:2]) )
        #snippet = snippet.reshape( (194, 257) )

        # reshape regenerated output into a spectrogram
        #regen = regen.detach().numpy().reshape( (194, 257) )

        #c = np.concatenate((snippet, whiteborder))
        #c = np.concatenate((c, regen))

        #plt.matshow(c)
        #plt.savefig(folder + "/regen-spectrograms/" + str(s))
        # close pyplot every time to save memory
        #plt.close()

