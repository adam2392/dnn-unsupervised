from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from ieeg_cnn_rnn import IEEGdnn

import argparse
def usegpu():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-o", "--output", required=True,
		help="path to output plot")
	ap.add_argument("-g", "--gpus", type=int, default=1,
		help="# of GPUs to use for training")
	args = vars(ap.parse_args())
	 
	# grab the number of GPUs and store it in a conveience variable
	G = args["gpus"]

if __name__ == '__main__':
	# Extract MNIST Dataset
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	# extract images and labels
	X_train= mnist.train.images
	y_train = mnist.train.labels
	X_test = mnist.test.images
	y_test = mnist.test.labels

	# set the image data paramters
	imsize = np.sqrt(X_train.shape[1])
	n_colors = 1

	# reshape into correct size to feed into NN
	X_train = np.reshape(X_train, [-1, imsize, imsize, n_colors])
	X_test = np.reshape(X_test, [-1, imsize, imsize, n_colors])

	print X_train.shape
	print y_train.shape

	# parameters for the layers
	n_classes = y_train.shape[1]
	n_fcunits = 1024
	w_init=None
	n_layers=(2,1)

	bsize=32
	nepochs=1000

	# parameters for training
	loss = keras.losses.categorical_crossentropy
	optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	metrics = ['accuracy']

	################### Instantiate the DNN For IEEG Object
	ieegdnnmodel = IEEGdnn(imsize=imsize, n_colors=n_colors)
	ieegdnnmodel.build_cnn(w_init=w_init, n_layers=(2,1))
	ieegdnnmodel.build_output(n_classes, n_fcunits)
	ieegdnnmodel.compile_model(loss, optimizer, metrics)

	# train
	ieegdnnmodel.train(X_train, y_train, batch_size=bsize, epochs=nepochs)
	
	# test and evaluate model
	ieegdnnmodel.eval(X_test, y_test, batch_size=bsize)



