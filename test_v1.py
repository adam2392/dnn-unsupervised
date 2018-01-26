import sys
sys.path.append('./dnn/')

import model.ieeg_cnn_rnn
import model.train

import processing.util as util

import keras
from keras.callbacks import ModelCheckpoint
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

if __name__ == '__main__':
	traindatadir = str(sys.argv[1])
	tempdatadir = str(sys.argv[2])

	sys.stdout.write(os.environ["CUDA_VISIBLE_DEVICES"])
	##################### PARAMETERS FOR NN ####################
	# image parameters
	imsize=32
	numfreqs = 5

	# layer parameters
	w_init = None
	n_layers = (4,2,1)
	poolsize = (2,2)
	filtersize = (3,3)

	size_fc = 1024
	DROPOUT = False #True
	# ieegdnn = model.ieeg_cnn_rnn.IEEGdnn(imsize=imsize, n_colors=numfreqs)