#!/usr/bin/env python
# coding: utf-8

'''
This program converts wav files into spectrograms and saves them as numpy arrays.

To define the root folder, open and edit the file
The first argument is wether the wav file is dual or mono channel. enter mono for mono, dual for dual
The second argument is the path relative to the root where the audio files are
The second argument is the path relative to the root + the audio file folder where the numpy matrices shall be saved

Example use:
sudo python3 spectrogram.py dual audio/ spectrograms/

Hunaid Hameed
hunaidhameed@hotmail.com
'''

from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from os import listdir, mkdir
import sys

def makeSpectrograms ():
	root = "/net/projects/scratch/winter/valid_until_31_July_2021/0-animal-communication/"
	#root = "/home/hunaid/lucidmonkeys/"
	path = root + sys.argv[2]
	
	filenames = [f for f in listdir(path)]
	print(filenames)
	
	savepath = root + sys.argv[2] + sys.argv[3]
	mkdir(savepath)
	
	for filename in filenames:
		sample_rate, samples = wavfile.read(path + filename)
		
		#samples = samples[:500]
		
		if sys.argv[1] == "mono":
			frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

			np.save(root + sys.argv[3] + filename)
			
		if sys.argv[1] == "dual":
			channels = np.array([ch1 for bothchannels in samples for ch1 in bothchannels])
			channel1 = channels[::1]
			channel2 = channels[::2]
			del channels
	
			frequencies, times, spectrogram = signal.spectrogram(channel1, sample_rate)
			np.save(savepath + filename[:-4] + "_channel1", spectrogram)
			
			frequencies, times, spectrogram = signal.spectrogram(channel1, sample_rate)
			np.save(savepath + filename[:-4] + "_channel2", spectrogram)

def main():
	makeSpectrograms()

if __name__ ==	"__main__":
	main()
