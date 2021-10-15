#!/usr/bin/env python3

"""
Module: clustering.py

Clustering for convolutional autoencoder :
* Iterate through the whole dataset(both training and validation) passing them
  through only encoder to get bottleneck vectors for each data point.
* Cluster these bottleneck vectors using unsupervised clustering methods like
  Kmeans or GaussianMixture modeling.

Online resources we find useful during implementation
Methods for selecting the optimal number of clusters: https://towardsdatascience.com/cheat-sheet-to-implementing-7-methods-for-selecting-optimal-number-of-clusters-in-python-898241e1d6ad
Gap statistics intro: https://glowingpython.blogspot.com/2019/01/a-visual-introduction-to-gap-statistics.html
KMeans, elbow and Silhouette: https://realpython.com/k-means-clustering-python/

Authors: Krupal, Rachael

"""

import os
from pathlib import Path

from math import ceil, floor
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
#local system hangs while processing GMM so commented out for now
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from kneed import KneeLocator # to identify the elbow point programmatically
import argparse

import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid

from models.residual_encoder import DefaultEncoderOpts
from models.residual_encoder import ResidualEncoder as Encoder
from models.residual_decoder import ResidualDecoder as Decoder
#from models.residual_encoder import DefaultEncoderOpts, ResidualEncoder as Encoder
from collections import OrderedDict
from data.audiodataset import DefaultSpecDatasetOps, StridedAudioDataset, get_audio_files_from_dir, Dataset
from utils.logging import Logger

parser = argparse.ArgumentParser()

parser.add_argument(
    "-d",
    "--debug",
    dest="debug",
    action="store_true",
    help="Log additional training and model information.",
)

""" Directory parameters """

parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    help="Path to a trained model.",
)

parser.add_argument(
    "--log_dir", type=str, default=None, help="The directory to store the logs."
)

parser.add_argument(
    "--clustering_dir",
    type=str,
    help="The directory where the clustering outputs will be stored.",
)

parser.add_argument(
    "--data_dir", # '/net/projects/scratch/winter/valid_until_31_July_2021/0-animal-communication/data_grid/Chimp_IvoryCoast/xxx'
    type=str,
    help="The path to the dataset directory.",
)

parser.add_argument(
    "--decod_dir",
    type=str,
    default=None,
    help="The directory to store the regenerated/decoded spectrograms.",
)

""" Clustering parameters """

parser.add_argument(
    "--calc_optimal_num_clusters", type=str, default=None, help="use elbow or gap statistics to find the optimal number of clusters",
)

parser.add_argument(
    "--max_clusters", type=int, default=None, help="The max number of clusters to try for calculating optimal number of clusters."
)

parser.add_argument(
    "--num_clusters", type=int, default=None, help="The number of clusters to cluster all the data points."
)

parser.add_argument(
    "--clustering_algorithm",
    type=str,
    default=None,
    help="Use either KMeans or GMM to cluster the embeddings",
)

""" Data input parameters """
parser.add_argument(
    "--sequence_len", type=float, default=2, help="Sequence length in [s]."
)

parser.add_argument(
    "--hop", type=float, default=1, help="Hop [s] of subsequent sequences."
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


ARGS = parser.parse_args()
ARGS.cuda = torch.cuda.is_available() and ARGS.cuda # check if both hardware and user's intention of using cuda
ARGS.device = torch.device("cuda") if ARGS.cuda else torch.device("cpu")

log = Logger("Clustering", ARGS.debug, ARGS.log_dir)

kmeans_kwargs = {
    "init": "k-means++",
    "n_init": 10, #  sets the number of initializations to perform
    "max_iter": 300,
    "random_state": 42,
}

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

def save_decod_spec(spec, epoch):
    #if not os.path.isdir(ARGS.decod_dir):
    #    os.makedirs(ARGS.decod_dir, exist_ok=True)
    spec = spec.view(spec.size(0), 1, 128, 256)
    save_image(spec, os.path.join(ARGS.decod_dir, 'reconstructed_epoch_{}.png'.format(epoch)))

def kmeans_optimalK(data, max_clusters=30):
    kmeans = [KMeans(
        n_clusters=i,
        **kmeans_kwargs,
    ) for i in range(1, max_clusters+1)]
    print("kmeans is ", kmeans)
    scores = [kmeans[i].fit(data).inertia_ for i in range(len(kmeans))]
    print("scores are ", scores)
    #x = range(1, max_clusters+1)
    #calculation = np.abs(np.diff(x).mean())
    #print(x)
    #print(calculation)
    #suggested_elbow = get_elbow(np.arange(1, max_clusters+1), scores)
    kl = KneeLocator(np.arange(1, max_clusters+1), scores)
    suggested_elbow = kl.elbow
    return scores, suggested_elbow

def get_elbow(max_clusters, scores, curve="convex", direction='decreasing'): #concave
    """Helper function to identify the elbow point programmatically"""
    kl = KneeLocator(
        max_clusters, scores, curve, direction
    )
    return kl.elbow

def elbow_plot(max_clusters, scores, path):
    plt.plot(np.arange(1, max_clusters+1), scores, linestyle='--', marker='o', color='b')
    plt.xlabel('Number of clusters')
    plt.ylabel('Score (negative of the K-means objective)')
    plt.title('Elbow curve with max_clusters={}'.format(max_clusters))
    plt.savefig(os.path.join(path, 'ElbowCurve.png'))
    log.info("The elbow plot is saved under directory {}".format(path))

def gap_optimalK(data, num_refs=4, max_clusters=15):
    """
    Calculate kMeans optimal number of clusters using Gap statistics
    adapted from
    https://datasciencelab.wordpress.com/tag/gap-statistic/
    https://towardsdatascience.com/cheat-sheet-to-implementing-7-methods-for-selecting-optimal-number-of-clusters-in-python-898241e1d6ad
    and https://glowingpython.blogspot.com/2019/01/a-visual-introduction-to-gap-statistics.html

    :param data: (n_samples, n_features)
    :param num_refs: number of sample reference datasets to create
    :param maxClusters: Maximum number of clusters to test for

    :return: (gaps, optimalK)

    """

    #if len(data.shape) == 1:
        #data = data.reshape(-1, 1)

    gaps = np.zeros((len(range(1, max_clusters+1)),))
    resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})

    for gap_index, k in enumerate(range(1, max_clusters+1)):
        # Holder for reference dispersion results
        ref_disps = np.zeros(num_refs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(num_refs):
            # Create new random reference set
            random_reference = np.random.random_sample(size=data.shape)

            # Fit to it
            km = KMeans(n_clusters=k, **kmeans_kwargs)
            km.fit(random_reference)

            ref_disp = km.inertia_
            ref_disps[i] = ref_disp

        # Fit cluster to original data and create dispersion
        km = KMeans(n_clusters=k, **kmeans_kwargs)
        km.fit(data)

        orig_disp = km.inertia_

        # Calculate gap statistic
        # online resources use: gap = np.log(np.mean(refDisps)) - np.log(origDisp). But it is believed to be wrong
        # because of the equation 3 in the original paper
        gap = np.mean(np.log(ref_disps)) - np.log(orig_disp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        resultsdf = resultsdf.append({'clusterCount': k, 'gap': gap}, ignore_index=True)
    return (gaps.argmax() + 1, resultsdf)

def gap_plot(df, path):
    print('clusterCount is ', df['clusterCount'])
    plt.plot(df['clusterCount'], df['gap'], linestyle='--', marker='o', color='b');
    plt.xlabel('K');
    plt.ylabel('Gap Statistic');
    plt.title('Gap Statistic vs. K');
    plt.savefig(os.path.join(path, 'GapStat.png'));
    log.info("The gap plot is saved under directory {}".format(path))



if __name__ == '__main__':

    audio_files = get_audio_files()
    #Path(ARGS.decod_dir).mkdir(parents=True, exist_ok=True)
    if ARGS.decod_dir is not None and os.path.isdir(ARGS.decod_dir):
        os.makedirs(ARGS.decod_dir, exist_ok=True)

    # load the trained model
    if ARGS.model_path is not None:
        model_dict = torch.load(ARGS.model_path)
        encoder = Encoder(model_dict["encoderOpts"]).to(ARGS.device)
        encoder.load_state_dict(model_dict["encoderState"])
        decoder = Decoder(model_dict["decoderOpts"]).to(ARGS.device)
        decoder.load_state_dict(model_dict["decoderState"])
        #model = encoder
        model = nn.Sequential(
            OrderedDict([("encoder", encoder), ("decoder", decoder)])
        )
        dataOpts = model_dict["dataOpts"]

    log.info(model)

    if torch.cuda.is_available() and ARGS.cuda:
        model = model.cuda()
    model.eval()

    sr = dataOpts["sr"] # modified, s.t. not hard-coded
    hop_length = dataOpts["hop_length"]
    n_fft = dataOpts["n_fft"]

    try:
        n_freq_bins = dataOpts["num_mels"]
    except KeyError:
        n_freq_bins = dataOpts["n_freq_bins"]

    freq_compression = dataOpts["freq_compression"] # added, missing in orig master; is freq compression not needed during inference?
    fmin = dataOpts["fmin"]
    fmax = dataOpts["fmax"]
    log.debug("dataOpts: " + str(dataOpts))
    #sequence_len = int(ceil(ARGS.sequence_len * sr))
    sequence_len = int(
        float(ARGS.sequence_len) / 1000 * dataOpts["sr"] / dataOpts["hop_length"]
    )
    hop = int(ceil(ARGS.hop * sr))

    log.info("Predicting {} files".format(len(audio_files)))

    #for file_name in audio_files:
    #    log.info(file_name)
    #    dataset = StridedAudioDataset(
    #        os.path.join(ARGS.data_dir, file_name.strip()),
    #        sequence_len=sequence_len,
    #        hop=hop,
    #        sr=sr,
    #        fft_size=n_fft,
    #        fft_hop=hop_length,
    #        n_freq_bins=n_freq_bins,
    #        freq_compression=freq_compression, # added
    #        f_min=fmin,
    #        f_max=fmax,
    #    )
    #    data_loader = torch.utils.data.DataLoader(
    #        dataset,
    #        batch_size=ARGS.batch_size,
    #        num_workers=ARGS.num_workers,
    #        pin_memory=True,
    #    )

    #    log.info("size of the file(samples)={}".format(dataset.n_frames))
    #    log.info("size of hop(samples)={}".format(hop))
    #    stop = int(max(floor(dataset.n_frames / hop), 1))
    #    log.info("stop time={}".format(stop))

    dataset = Dataset(
        file_names=audio_files,
        working_dir=ARGS.data_dir,
        #cache_dir=ARGS.cache_dir,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_freq_bins=n_freq_bins,
        freq_compression=freq_compression,
        f_min=fmin,
        f_max=fmax,
        seq_len=sequence_len,
        augmentation=None,
        noise_files=None,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=ARGS.batch_size, # default: 1
        num_workers=ARGS.num_workers,
        pin_memory=True,
        )

    #log.info("size of the file(samples)={}".format(dataset.n_frames))
    #log.info("size of hop(samples)={}".format(hop))
    #stop = int(max(floor(dataset.n_frames / hop), 1))
    #log.info("stop time={}".format(stop))

    # collect embeddings for all data points
    file_names = []
    bottleneck_outputs = []

    with torch.no_grad():
        for i, (input_specs, label) in enumerate(data_loader):
            #print(input_specs)
            # remove file path to have only file name, ex : ['path/to/directory/file_1.wav']
            file_name = str(label['file_name'])[::-1]  # reverse string
            file_name = file_name.split("/")[0]
            file_name = file_name[::-1].split("'")[0]  # align it back to the right order
            file_names.append(file_name)
            # log.info("File-name : {}".format(file_name))

            input_specs = input_specs.to(device=ARGS.device)
            bottleneck_output = model.encoder(input_specs)
            bottleneck_output = bottleneck_output.cpu()

            #print("bottleneck_output .shape : ", bottleneck_output.shape)
            #print("bottleneck_output[0].shape :", bottleneck_output[0].shape)
            # uncomment if wanting to visualize bottleneck features
            #img = bottleneck_output[0]
            #save_image(img, os.path.join(ARGS.decod_dir, 'encoder_bottleneck_{}.png'.format(i)))

            bottleneck_output = np.reshape(bottleneck_output.detach().numpy(), newshape=(-1))
            # bottleneck_output = torch.flatten(bottleneck_output)
            # bottleneck_output = bottleneck_output.detach().numpy()
            bottleneck_outputs.append(bottleneck_output)

    log.debug("Finished extract code")
    log.debug(kmeans_kwargs)

    # calculate the optimal number of clusters using elbow or gap statistics
    if ARGS.calc_optimal_num_clusters is not None:

        if ARGS.calc_optimal_num_clusters == 'elbow':
            """ Finding optimal number of clusters, by using elbow curve (below are default values)"""
            scores, suggested_elbow = kmeans_optimalK(bottleneck_outputs, max_clusters=ARGS.max_clusters)
            log.info("The suggested optimal K is {}".format(suggested_elbow))
            elbow_plot(ARGS.max_clusters, scores, ARGS.clustering_dir)
        elif ARGS.calc_optimal_num_clusters == 'gap':
            score_g, df = gap_optimalK(np.array(bottleneck_outputs), num_refs=5, max_clusters=ARGS.max_clusters)
            gap_plot(df, ARGS.clustering_dir)
            log.info("The suggested optimal K is {}".format(score_g))

    # if you already know how many clusters you would like to have,
    # you could directly train the clustering model and cluster the embeddings
    if ARGS.num_clusters is not None and ARGS.clustering_algorithm is not None:

        if ARGS.clustering_algorithm == "kmeans":
            kmeans = KMeans(n_clusters=ARGS.num_clusters, **kmeans_kwargs)
            pred_kmeans = kmeans.fit_predict(bottleneck_outputs)

            log.info("predictions : {}".format(pred_kmeans))
            print("Cluster centers of Kmeans : ", kmeans.cluster_centers_)

            df = pd.DataFrame(columns=["filename"] + ["cluster_number"])

            # print file names with respective cluster numbers
            for i in range(len(data_loader)):
                log.info("file name : {}, predicted cluster - Kmeans : {}".format(file_names[i], pred_kmeans[i]))
                # log.info("file name : {}, predicted cluster - GaussianMixture : {}".format(file_names[i], pred_gm[i]))

                df = df.append(dict(zip(df.columns, [file_names[i]] + [pred_kmeans[i]])), ignore_index=True)

            if ARGS.clustering_dir is not None: 
                if Path(ARGS.clustering_dir).exists() == False:
                    os.mkdir(ARGS.clustering_dir)
                    
                df.to_csv(ARGS.clustering_dir + "/Kmeans_clusters.csv")
                log.info("kmeans_clusters csv is saved under directory {}".format(ARGS.clustering_dir))


        elif ARGS.clustering_algorithm == "gmm":
            gm = GaussianMixture(n_components=2, random_state=0)
            pred_gm = gm.fit_predict(bottleneck_outputs)

            log.info("predictions : {}".format(pred_gm))
            #print("Cluster centers of GaussianMixture : {:.8f}".format(gm.means_))

            df = pd.DataFrame(columns=["filename"] + ["cluster_number"])

            # print file names with respective cluster numbers
            for i in range(len(data_loader)):
                log.info("file name : {}, predicted cluster - GaussianMixture : {}".format(file_names[i], pred_gm[i]))
                df = df.append(dict(zip(df.columns, [file_names[i]] + [pred_gm[i]])), ignore_index=True)

            if ARGS.clustering_dir is not None:
                if Path(ARGS.clustering_dir).exists() == False:
                    os.mkdir(ARGS.clustering_dir)
                                
                df.to_csv(ARGS.clustering_dir + "/gmm_clusters.csv")
                log.info("gmm_clusters csv is saved under directory {}".format(ARGS.clustering_dir))

        else:
            log.error("Pls choose a clustering algorithm - kmeans or gmm (in a case sensitive manner)")

    #summary_dir = ARGS.clustering_dir
    #if summary_dir is not None:
        #df.to_csv(summary_dir + "/Kmeans_clusters")

    #print("km.cluster_centers_ length :", len(km.cluster_centers_))

"""
    if ARGS.decod_dir is not None:
        bottleneck_output = 0
        with torch.no_grad():
            for i in range(len(bottleneck_outputs)):
                bottleneck_output = torch.tensor(bottleneck_outputs[i]).to(ARGS.device)
                #print("Krupal : bottleneck_output shape :", bottleneck_output.shape)
                bottleneck_output = torch.reshape(bottleneck_output, (-1, 4, 4, 8)) #512

                #print("bottleneck_output .shape : ", bottleneck_output.shape)
                #print("bottleneck_output[0].shape :", bottleneck_output[0].shape)
                img = bottleneck_output[0]
                save_image(img, os.path.join(ARGS.decod_dir, 'decoder_bottleneck_{}.png'.format(i)))

                # bottleneck_output = torch.reshape(bottleneck_output, (1, 512, 4, 8))
                # not valid bottleneck_output = torch.unflatten(bottleneck_output, (1, 512, 4, 8))
                # bottleneck_output = bottleneck_output.unflatten(-1, (512, 4, 8))
                regenerate_spec = model.decoder(bottleneck_output)
                # Krupal :
                #print("Decoder's regenerated spec from embeddings' shape :", regenerate_spec.shape)
                epoch = "regenerated_spec_" + str(i)
                save_decod_spec(regenerate_spec.cpu().data, epoch)  # change "cpu' to device

    log.close()
"""




# -------------------------- leftover from Lucidmonkey --------------------------
# extract features
#df = pd.DataFrame( columns = ["filename"] + ["bottleneck" + str(_) for _ in range(nbottleneck)] )
#for spectrogram in data:
    #imagename = spectrogram["name"]
    #spectrogram = spectrogram["spectrogram"]

    # if the spectrogram is not of width 194 units, don't run the iteration
    #if spectrogram.shape[1] != 194:
        #continue

    #norm = np.linalg.norm(spectrogram)
    #snippet = spectrogram / norm
    #snippet = torch.reshape( snippet, (-1,) ).to(device)

    #features = model.generatefeatures(snippet).detach().numpy()
    #features = [str(f.item()) for f in features]
    #df = df.append( dict( zip( df.columns, [imagename] + list(features) ) ), ignore_index=True )

#df.to_csv(folder + "/features")

# perform clustering
#clustering = GaussianMixture(n_components=7, random_state=0).fit(df[[ "bottleneck" + str(_) for _ in range(nbottleneck) ]])

#df["clusterlabels"] = clustering.means_

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#r = [-1,1]
#X, Y = np.meshgrid(r, r)
#ax.scatter3D(points[:, 0], points[:, 1], points[:, 2])
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')
#plt.savefig(folder + '/gmmmixtures')

# write to file
#df.to_csv(folder + "/features")

#sys.stdout.close()
#sys.stderr.close()
