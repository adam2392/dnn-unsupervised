import time

import numpy as np

from functools import reduce
import math as m

### Necessary libraries for interfacing with Data
import scipy.io
from scipy.interpolate import griddata
from sklearn.preprocessing import scale

######### import DNN frameworks #########
import tensorflow as tf
import keras

######### import DNN for training using GPUs #########
from keras.utils.training_utils import multi_gpu_model

# utility functionality for keras - preprocessing sequential data
from keras.preprocessing import sequence 
# preprocessing - image data
from keras.preprocessing.image import ImageDataGenerator


np.random.seed(1234)

def train(model,xtrain, ytrain, batch_size=32, epochs=10):
    '''
    Main function to train the DNN model constructed:

    Things to Add:
    1. shuffling of minibatches
    2.  
    '''
    model.fit(xtrain, ytrain, 
    	verbose=1, 
    	batch_size=batch_size, 
    	epochs=epochs)

def eval(model, xtest, ytest, batch_size=32):
	score = model.evaluate(xtest, ytest, batch_size=batch_size)

    acc_train_history = score.history['acc']
    acc_test_history = score.history['val_acc']
    loss_train_history = score.history['loss']
    loss_test_history = score.history['val_loss']

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """
    Iterates over the samples returing batches of size batchsize.
    :param inputs: input data array. It should be a 4D numpy array for images 
    [n_samples, n_colors, W, H] and 

    5D numpy array if working with sequence of images 
    [n_timewindows, n_samples, n_colors, W, H].

    :param targets: vector of target labels. (0, or 1 for seizure or non seizure)
    :param batchsize: Batch size
    :param shuffle: Flag whether to shuffle the samples before iterating or not.
    :return: images and labels for a batch
    """
    if inputs.ndim==4:
        input_len = inputs.shape[0]
    elif inputs.ndim == 5:
        input_len = inputs.shape[1]
    assert input_len == len(targets)

    if shuffle:
        indices = np.arange(input_len)
        np.random.shuffle(indices)

    # Create generator objects iterating over the inputs with batchsize
    for start_idx in range(0, input_len, batchsize):
    	# get the 'excerpt' of data to be used right now
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        # return the generator of inputs & target class
        if inputs.ndim == 4:
            yield inputs[excerpt], targets[excerpt]
        elif inputs.ndim == 5:
            yield inputs[:, excerpt], targets[excerpt]

def reformatInput(data, labels, indices):
    """
    Receives the the indices for train and test datasets.
    Outputs the train, validation, and test data and label datasets.
    """

    trainIndices = indices[0][len(indices[1]):]
    validIndices = indices[0][:len(indices[1])]
    testIndices = indices[1]
    # Shuffling training data
    # shuffledIndices = np.random.permutation(len(trainIndices))
    # trainIndices = trainIndices[shuffledIndices]
    if data.ndim == 4:
        return [(data[trainIndices], np.squeeze(labels[trainIndices]).astype(np.int32)),
                (data[validIndices], np.squeeze(labels[validIndices]).astype(np.int32)),
                (data[testIndices], np.squeeze(labels[testIndices]).astype(np.int32))]
    elif data.ndim == 5:
        return [(data[:, trainIndices], np.squeeze(labels[trainIndices]).astype(np.int32)),
                (data[:, validIndices], np.squeeze(labels[validIndices]).astype(np.int32)),
                (data[:, testIndices], np.squeeze(labels[testIndices]).astype(np.int32))]


def train(model, images, labels, fold,
	batch_size=32, num_epochs=5):
    """
    A sample training function which loops over the training set and evaluates the network
    on the validation set after each epoch. Evaluates the network on the training set
    whenever the
    :param images: input images
    :param labels: target labels
    :param fold: tuple of (train, test) index numbers
    :param model_type: model type ('cnn', '1dconv', 'maxpool', 'lstm', 'mix')
    :param batch_size: batch size for training
    :param num_epochs: number of epochs of dataset to go over for training
    :return: none
    """
    num_classes = len(np.unique(labels))
    
    # get train set and test set using utility reformatting function
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = reformatInput(images, labels, fold)
    X_train = X_train.astype("float32", casting='unsafe')
    X_val = X_val.astype("float32", casting='unsafe')
    X_test = X_test.astype("float32", casting='unsafe')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
 
    # ADAM
	ADAM = keras.optimizers.Adam(lr=0.001, 
		beta_1=0.9, beta_2=0.999, 
		epsilon=1e-08, decay=0.0)
	model.compile(loss='categorical_crossentropy', 
		optimizer=ADAM, 
		metrics=["accuracy"])

	# the set of callbacks
	aug = ImageDataGenerator(width_shift_range=0.1,
	    height_shift_range=0.1, horizontal_flip=True,
	    fill_mode="nearest")
	callbacks = [LearningRateScheduler(poly_decay)]
	INIT_LR = 5e-3

	HH = model.fit_generator(
	    aug.flow(X_train, y_train, batch_size=64 * G), # adds augmentation to data using generator
	    validation_data=(X_test, y_test),  
	    steps_per_epoch=len(X_train) // (64 * G),    #
	    epochs=NUM_EPOCHS,
	    callbacks=callbacks, verbose=2)

	return HH

