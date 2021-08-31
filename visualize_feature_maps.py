#!/usr/bin/env python3

import os
import json
import math
from math import ceil
import pathlib
import argparse
import utils.metrics as m
import matplotlib.pyplot as plt
#import cv2 as cv #to read the image

import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim

from data.audiodataset import (
    #get_audio_files_from_dir,
    #get_broken_audio_files,
    #DatabaseCsvSplit,
    #DefaultSpecDatasetOps,
    #Dataset,
    StridedAudioDatasetVis
)

from trainer import Trainer
from utils.logging import Logger
from collections import OrderedDict
from models.residual_encoder import DefaultEncoderOpts
from models.residual_encoder import ResidualEncoder as Encoder
from models.classifier import Classifier, DefaultClassifierOpts
import data.transforms as T
from utils.FileIO import AsyncFileReader

parser = argparse.ArgumentParser()

#parser.add_argument(
#    "-d",
#    "--debug",
#    dest="debug",
#    action="store_true",
#    help="Print additional training and model information.",
#)

parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    help="Path to a model.",
)

#parser.add_argument(
#    "--checkpoint_path",
#    type=str,
#    default=None,
#    help="Path to a checkpoint. "
#    "If provided the checkpoint will be used instead of the model.",
#)

#parser.add_argument(
#    "--log_dir", type=str, default=None, help="The directory to store the logs."
#)

#parser.add_argument(
#    "--cache_dir",
#    type=str,
#    help="The path to the cached spec directory.",
#)

#parser.add_argument(
#    "--sequence_len", type=float, default=2, help="Sequence length in [s]."
#)

#parser.add_argument(
#    "--hop", type=float, default=1, help="Hop [s] of subsequent sequences."
#)

#parser.add_argument(
#    "--threshold",
#    type=float,
#    default=0.5,
#    help="Threshold for the probability for detecting an orca.",
#)

#parser.add_argument(
#    "--batch_size", type=int, default=1, help="The number of images per batch."
#)

#parser.add_argument(
#    "--num_workers", type=int, default=4, help="Number of workers used in data-loading"
#)

#parser.add_argument(
#    "--no_cuda",
#    dest="cuda",
#    action="store_false",
#    help="Do not use cuda to train model.",
#)

#parser.add_argument(
#    "audio_files", type=str, nargs="+", help="Audio file to predict the call locations"
#)

#parser.add_argument(
#    ""
#)

# execute parse_args()
ARGS = parser.parse_args()

#log = Logger("PREDICT", ARGS.debug, ARGS.log_dir)

if __name__ == "__main__":
    # torch.load(): load the dictionary locally, to deserialize the dictionary
    model_dict = torch.load(ARGS.model_path)
    encoder = Encoder(model_dict["encoderOpts"])
    # load_state_dict() function takes a dictionary object, NOT a path to a saved object;
    encoder.load_state_dict(model_dict["encoderState"])
    classifier = Classifier(model_dict["classifierOpts"])
    classifier.load_state_dict(model_dict["classifierState"])
    model = nn.Sequential(
            OrderedDict([("encoder", encoder), ("classifier", classifier)])
    )
    dataOpts = model_dict["dataOpts"]

    #log.info(model)
    #print(model)
    #model.summary()

    #if torch.cuda.is_available() and ARGS.cuda:
    #    model = model.cuda()
    model.eval()

    model_weights = []
    conv_layers = []

    model_modules = list(model.modules())
    #print(model_children)
    # print('model children saved as a list')

    counter = 0
    for i in range(len(model_modules)):
        if type(model_modules[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_modules[i].weight)
            conv_layers.append(model_modules[i])
    print(f"Total conv layers: {counter}")
    print(conv_layers)

    #plt.figure(figsize=(20,17))
    #for i, filter in enumerate(model_weights[0]):
        #plt.subplot(8, 8, i+1)
        #plt.imshow(filter[0, :, :].detach(), cmap='gray')
        #plt.axis('off')
        #plt.savefig('/mnt/2ndSSD/Osnabrueck/SP/interpretability/orca_visualization/filter.png')

#    sr = 44100
#    hop_length = dataOpts["hop_length"]
#    n_fft = dataOpts["n_fft"]

#    try:
#        n_freq_bins = dataOpts["num_mels"]
#    except KeyError:
#        n_freq_bins = dataOpts["n_freq_bins"]

    #freq_compression = dataOpts["freq_compression"]
    #fmin = dataOpts["fmin"]
    #fmax = dataOpts["fmax"]
    #log.debug("dataOpts: " + str(dataOpts))
    #sequence_len = int(ceil(ARGS.sequence_len * sr))
    #hop = int(ceil(ARGS.hop * sr))
    #cache_dir = ARGS.cache_dir

    #input = torch.load("/mnt/2ndSSD/Osnabrueck/SP/interpretability/cache/N7_4127_1993_088A_179188_180670")
    # the .spec is not from the cache created during training. But the input returned by dataloader
    input = torch.load("/home/am/Desktop/EVA_Chimps/networks_visualization/detector/call-chimp-rom-ab-sm2-bout_6993_2012_1009_0_2000_cache.spec")
    print("input shape is ", input)

    print("start passing input through layers")
    results = [conv_layers[0](input)]
    residuals = []
    #print(results[-1].size())
    for i in range(1, len(conv_layers)):
        if i == 5 or i == 10 or i == 15:
            y = conv_layers[i](results[-1])
            residuals.append(y)
            print(y.size())
            print(f"passed through {i} - residual connection")
        elif i == 9 or i == 14 or i == 19:
            result = conv_layers[i](results[-1]) + residuals[-1]
            results.append(result)
            print(result.size())
            print("residuals added")
        else:
            results.append(conv_layers[i](results[-1]))
            #print(result.size())
            print(f'passed through {i} conv layer')

    outputs = results
    print("outputs collected")


    print("preparing visualization")
    print("-----------------------")
    for num_layer in range(len(outputs)):
        plt.figure(figsize=(30, 30))
        layer_viz = outputs[num_layer][0, :, :, :]
        layer_viz = layer_viz.data
        print(layer_viz.size())
        for i, filter in enumerate(layer_viz):
            if i == 64:    # only visualize 64 feature maps from each layer
                break
            plt.subplot(8, 8, i + 1)
            plt.imshow(filter, cmap='gray')
            plt.axis("off")
        if num_layer in [5,6,7,8]:
            num_layer += 1
        elif num_layer in [9,10,11,12]:
            num_layer += 2
        elif num_layer in [13,14,15,16]:
            num_layer += 3
        print(f"Saving layer {num_layer} feature maps ...")
        plt.savefig(f"/home/am/Desktop/EVA_Chimps/networks_visualization/detector/layer_{num_layer}.png")
        plt.close()

