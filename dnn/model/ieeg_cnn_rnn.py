import time

import numpy as np
np.random.seed(1234)
from functools import reduce
import math as m

### Necessary libraries for interfacing with Data
import scipy.io
from scipy.interpolate import griddata
from sklearn.preprocessing import scale

######### import DNN for training using GPUs #########
from keras.utils.training_utils import multi_gpu_model

######### import DNN frameworks #########
import tensorflow as tf
import keras

# import high level optimizers, models and layers
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import InputLayer

# for CNN
from keras.layers import Conv2D, MaxPooling2D
# for RNN
from keras.layers import LSTM
# for general NN behavior
from keras.layers import Dense, Dropout, Flatten

# utility functionality for keras
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding

# preprocessing
from keras.preprocessing.image import ImageDataGenerator
# from utils import augment_EEG, cart2sph, pol2cart
# from keras import backend as K

class IEEGdnn():
    def __init__(self, imsize=32, n_colors=3):
        # initialize class elements
        self.imsize = imsize
        self.n_colors = n_colors
        self.model = Sequential()

    def build_cnn(self, w_init=None, n_layers=(4,2,1),poolsize=(2,2),n_filters_first=32):    
        DEBUG=0
        
        # check for weight initialization -> apply Glorotuniform
        if w_init is None:
            w_init = [keras.initializers.glorot_uniform()] * sum(n_layers)

        # model.add(InputLayer(input_shape=(self.imsize, self.imsize, self.n_colors)))
        # initialize counter
        count=0
        
        # add layers in VGG style
        for idx, n_layer in enumerate(n_layers):
            for ilay in range(n_layer):
                self.model.add(Conv2D(n_filters_first*(2 ** idx), 
                                 (3, 3),
                                 input_shape=(self.imsize, self.imsize, self.n_colors),
                                 kernel_initializer=w_init[count], activation='relu'))
                if DEBUG:
                    print self.model.output_shape
                    print idx, " and ", ilay
                count+=1

            # create a network at the end with a max pooling
            self.model.add(MaxPooling2D(pool_size=poolsize))
        return self.model

    def build_output(self, n_outunits, n_fcunits=1024):
        self.model.add(Flatten())
        self.model.add(Dense(n_fcunits, activation='relu'))
        self.model.add(Dense(n_outunits, activation='softmax'))

    '''
    Functions for completing and running the entire model
    '''
    def compile_model(self, loss, optimizer, metrics=['accuracy']):
        optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        
        # store the final configuration of the model
        self.model_config = self.model.get_config()

    def train(self, xtrain, ytrain, batch_size=32, epochs=10):
        '''
        Main function to train the DNN model constructed:

        Things to Add:
        1. shuffling of minibatches
        2.  
        '''
        self.model.fit(xtrain, ytrain, verbose=1, batch_size=batch_size, epochs=epochs)
    
    def eval(self, xtest, ytest, batch_size=32):
        self.score = self.model.evaluate(xtest, ytest, batch_size=batch_size)

        acc_train_history = self.score.history['acc']
        acc_test_history = self.score.history['val_acc']
        loss_train_history = self.score.history['loss']
        loss_test_history = self.score.history['val_loss']

    def build_cnn_lstm(self, input_vars, nb_classes,grad_clip=110, imsize=32, n_colors=3, n_timewin=3):
        '''
        Builds the complete network with LSTM layer to integrate time from sequences of EEG images.

        :param input_vars: list of EEG images (one image per time window)
        :param nb_classes: number of classes
        :param grad_clip:  the gradient messages are clipped to the given value during
                            the backward pass.
        :param imsize: size of the input image (assumes a square input)
        :param n_colors: number of color channels in the image
        :param n_timewin: number of time windows in the snippet
        :return: a pointer to the output of last layer
        '''
        # create a list of convnets in time
        convnets = []

        # initialize weights with nothing
        w_init = None

        # build parallel CNNs with shared weights across time
        for i in range(n_timewin):
            if i == 0:
                convnet, w_init = build_cnn(input_vars[i], imsize=imsize, n_colors=n_colors)
            else:
                convnet, _ = build_cnn(input_vars[i], w_init=w_init, imsize=imsize, n_colors=n_colors)
            convnets.append(FlattenLayer(convnet))

        # at this point convnets shape is [numTimeWin][n_samples, features]
        # we want the shape to be [n_samples, features, numTimeWin]
        # convpool = ConcatLayer(convnets)
        # convpool = ReshapeLayer(convpool, ([0], n_timewin, get_output_shape(convnets[0])[1]))
        # Input to LSTM should have the shape as (batch size, SEQ_LENGTH, num_features)
        convpool = LSTMLayer(convpool, num_units=128, grad_clipping=grad_clip,
            nonlinearity=lasagne.nonlinearities.tanh)
        # We only need the final prediction, we isolate that quantity and feed it
        # to the next layer.
        convpool = SliceLayer(convpool, -1, 1)      # Selecting the last prediction
        # A fully-connected layer of 256 units with 50% dropout on its inputs:
        convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
                num_units=256, nonlinearity=lasagne.nonlinearities.rectify)
        # And, finally, the output layer with 50% dropout on its inputs:
        convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
                num_units=nb_classes, nonlinearity=lasagne.nonlinearities.softmax)
        return convpool
