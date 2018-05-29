from .base import BaseNet
############################ NUM PROCESSING FXNS ############################
import numpy as np
############################ UTILITY FUNCTIONS ############################
import time
import math as m
############################ ANN FUNCTIONS ############################
######### import DNN for training using GPUs #########
# from keras.utils.training_utils import multi_gpu_model
######### import DNN frameworks #########
import tensorflow as tf
import keras
# import high level optimizers, models and layers
from keras.models import Sequential, Model
from keras.layers import InputLayer
# for CNN
from keras.layers import Conv1D, Conv2D, Conv3D, MaxPooling1D, MaxPooling2D, MaxPooling3D
# for general NN behavior
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Input, Concatenate, Permute, Reshape, Merge

# utility functionality for keras
from keras.layers.embeddings import Embedding
import pprint


class CNNFragility(BaseNet):
    def __init__(self, numwins, imsize=30, n_colors=1, num_classes=2, DROPOUT=True):
        '''
        To build an auxiliary type network with a 1D CNN going over the channel of interest,
        while a 2D CNN analyzes the PCA of the fragility network.


        Parameters:
        num_classes         (int) the number of classes in prediction space
        '''
        self.numwins = numwins

        # initialize class elements
        self.imsize = imsize
        self.n_colors = n_colors
        self.num_classes = num_classes
        self.DROPOUT = DROPOUT

        # start off with a relatively simple sequential model
        self.model = Sequential()

    def summaryinfo(self):
        summary = {
            'imsize': self.imsize,
            'numchans': self.n_colors,
            'numclasses': self.num_classes,
            'DROPOUT': self.DROPOUT,
            'w_init': self.w_init,
            'numlayers': self.n_layers,
            'numfilters': self.numfilters,
            'poolingsize': self.poolsize,
            'filteringsize': self.filter_size
        }
        pprint.pprint(summary)

    def buildmodel(self):
        w_init = None                       # weight initialization
        # number of convolutions per layer
        n_layers = (4, 2, 1)
        numfilters = 32                     # number of filters in first layer of each new layer
        poolsize = (2,)      # pool size
        filter_size = (3,)   # filter size

        self.w_init = w_init
        self.n_layers = n_layers
        self.numfilters = numfilters
        self.poolsize = poolsize
        self.filter_size = filter_size

        # build the two CNNs we want
        self._build_1dcnn(w_init=w_init,
                          n_layers=n_layers,
                          poolsize=poolsize,
                          n_filters_first=numfilters,
                          filter_size=filter_size)

        poolsize = (2, 4)      # pool size
        filter_size = (2, 6)   # filter size
        self._build_2dcnn(w_init=w_init,
                          n_layers=n_layers,
                          poolsize=poolsize,
                          n_filters_first=numfilters,
                          filter_size=filter_size)
        self._combinenets()

    def _build_1dcnn(self, w_init=None, n_layers=(4, 2, 1), poolsize=(3,), n_filters_first=32, filter_size=(3,)):
        '''
        Creates a Convolutional Neural network in VGG-16 style. It requires self
        to initialize a sequential model first.

        Parameters:
        w_init              (list) of all the weights (#layers * #nodes_in_layers)
        n_layers            (tuple) of number of nodes in each layer
        poolsize            (tuple) for max pooling poolsize along each dimension 
                            (e.g. 2D is (2,2) for pooling over 2 pixels in x and y direction)
        n_filters_first     (int) number of filters in the first layer
        filter_size         (int/tuple/list) the kernel/filter size of the width/height of
                            2D convolution window

        Returns:
        model               the sequential model object with all layers added in CNN style
        '''
        # check for weight initialization -> apply Glorotuniform
        if w_init is None:
            w_init = [keras.initializers.glorot_uniform()] * sum(n_layers)

        # set up input layer of CNN
        self.model1d = Sequential()
        self.model1d.add(InputLayer(input_shape=(self.numwins, self.n_colors)))
        # self.model1d.add(InputLayer(input_shape=(None, self.n_colors)))
        # initialize counter to keep track of which weight to assign
        count = 0
        # add the rest of the hidden layers
        for idx, n_layer in enumerate(n_layers):
            for ilay in range(n_layer):
                self.model1d.add(Conv1D(n_filters_first*(2 ** idx),
                                        kernel_size=filter_size,
                                        # input_shape=(self.numwins, self.n_colors),
                                        kernel_initializer=w_init[count],
                                        activation='relu'))
                # increment counter to the next weight initializer
                count += 1
            # create a network at the end with a max pooling
            self.model1d.add(MaxPooling1D(pool_size=poolsize))
        self.model1d.add(Flatten())

    def _combinenets(self):
        numfc = 512

        # define the two inputs (one is 1D, the other 2D)
        main_input = Input(shape=(self.numwins,),
                           dtype='float32', name='main_input')
        auxiliary_input = Input(shape=(self.imsize, self.imsize),
                                dtype='float32', name='aux_input')

        # create the two models, so that we can concatenate them
        model1d = Model(inputs=self.model1d.input, outputs=self.model1d.output)
        # flat1doutput = Dense(numfc)(model1d.output)
        # flat1doutput = Reshape((None,...))(flat1doutput)
        model2d = Model(inputs=self.model2d.input, outputs=self.model2d.output)
        # flat2doutput = Dense(numfc)(model2d.output)
        # flat2doutput = Reshape((None,...))(flat2doutput)

        # concatenate the two outputs
        x = keras.layers.concatenate([model1d.output, model2d.output])
        # x = keras.layers.concatenate([flat1doutput, flat2doutput])

        # build and stack the new merged layers with densely connected network
        x = Dense(numfc, activation='relu')(x)
        if self.DROPOUT:
            x = Dropout(0.5)(x)
        x = Dense(numfc, activation='relu')(x)
        if self.DROPOUT:
            x = Dropout(0.5)(x)
        x = Dense(numfc, activation='relu')(x)
        if self.DROPOUT:
            x = Dropout(0.5)(x)

        # And finally we add the main softmax regression layer
        main_output = Dense(self.num_classes, activation='softmax')(x)
        self.model = Model(
            inputs=[self.model1d.input, self.model2d.input], outputs=main_output)

    def _build_2dcnn(self, w_init=None, n_layers=(4, 2, 1), poolsize=(2, 2), n_filters_first=32, filter_size=(3, 3)):
        '''
        Creates a Convolutional Neural network in VGG-16 style. It requires self
        to initialize a sequential model first.

        Parameters:
        w_init              (list) of all the weights (#layers * #nodes_in_layers)
        n_layers            (tuple) of number of nodes in each layer
        poolsize            (tuple) for max pooling poolsize along each dimension 
                            (e.g. 2D is (2,2) for pooling over 2 pixels in x and y direction)
        n_filters_first     (int) number of filters in the first layer
        filter_size         (int/tuple/list) the kernel/filter size of the width/height of
                            2D convolution window

        Returns:
        model               the sequential model object with all layers added in CNN style
        '''
        # check for weight initialization -> apply Glorotuniform
        if w_init is None:
            w_init = [keras.initializers.glorot_uniform()] * sum(n_layers)
        # set up input layer of CNN
        self.model2d = Sequential()
        self.model2d.add(InputLayer(input_shape=(
            self.imsize, self.numwins, self.n_colors)))
        # initialize counter to keep track of which weight to assign
        count = 0
        # add the rest of the hidden layers
        for idx, n_layer in enumerate(n_layers):
            for ilay in range(n_layer):
                self.model2d.add(Conv2D(n_filters_first*(2 ** idx),
                                        kernel_size=filter_size,
                                        kernel_initializer=w_init[count],
                                        activation='relu'))
                # increment counter to the next weight initializer
                count += 1
            # create a network at the end with a max pooling
            self.model2d.add(MaxPooling2D(pool_size=poolsize))
        self.model2d.add(Flatten())
