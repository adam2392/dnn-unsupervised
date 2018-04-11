from .base import BaseNet
############################ NUM PROCESSING FXNS ############################
import numpy as np
############################ UTILITY FUNCTIONS ############################
import time
import math as m

############################ ANN FUNCTIONS ############################
######### import DNN frameworks #########
import tensorflow as tf
import keras

from keras import Model
# import high level optimizers, models and layers
from keras.models import Sequential, Model
from keras.layers import InputLayer

from keras.optimizers import Adam
# for CNN
from keras.layers import Conv1D, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, MaxPooling1D
from keras.layers import AveragePooling1D, AveragePooling2D
# for general NN behavior
from keras.layers import Dense, Dropout, Flatten, LeakyReLU
from keras.layers import Input, Concatenate, Permute, Reshape, Merge

# utility functionality for keras
# from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
import pprint


class iEEGCNN(BaseNet):
    def __init__(self, imsize=32, n_colors=3, num_classes=2, modeldim=2, DROPOUT=True):
        '''
        Parameters:
        num_classes         (int) the number of classes in prediction space
        '''
        # initialize class elements
        self.imsize = imsize
        self.n_colors = n_colors
        self.num_classes = num_classes
        self.modeldim = modeldim
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

    # def configure(self):
    #     # initialize loss function, SGD optimizer and metrics
    #     loss = 'binary_crossentropy'
    #     optimizer = Adam(lr=1e-4,
    #                                     beta_1=0.9,
    #                                     beta_2=0.999,
    #                                     epsilon=1e-08,
    #                                     decay=0.0)
    #     metrics = ['accuracy']
    #     self.modelconfig = self.model.compile(loss=loss,
    #                                             optimizer=optimizer,
    #                                             metrics=metrics)

    def buildmodel(self, output=True):
        w_init = None                       # weight initialization
        # number of convolutions per layer
        n_layers = (4, 3, 1)
        numfilters = 32                     # number of filters in first layer of each new layer
        # poolsize = ((2,)*self.modeldim)      # pool size
        # filter_size = ((3,)*self.modeldim)   # filter size
        poolsize = ((2,)*self.modeldim)      # pool size
        filter_size = ((2,)*self.modeldim)   # filter size

        size_fc = 1024

        self.w_init = w_init
        self.n_layers = n_layers
        self.numfilters = numfilters
        self.poolsize = poolsize
        self.filter_size = filter_size

        if self.modeldim == 1:
            self.build_vgg_1dcnn(w_init=w_init,
                              n_layers=n_layers,
                              poolsize=poolsize,
                              n_filters_first=numfilters,
                              filter_size=filter_size)
        elif self.modeldim == 2:
            self.build_vgg_2dcnn(w_init=w_init,
                              n_layers=n_layers,
                              poolsize=poolsize,
                              n_filters_first=numfilters,
                              filter_size=filter_size)
        elif self.modeldim == 3:
            self.build_vgg_3dcnn(w_init=w_init,
                              n_layers=n_layers,
                              poolsize=poolsize,
                              n_filters_first=numfilters,
                              filter_size=filter_size)
        else:
            raise ValueError(
                'Model dimension besides (1,2,3) not implemented!')

        if output:
            self.buildoutput(size_fc)

    def buildoutput(self, size_fc):
        self._build_seq_output(size_fc=size_fc)

    def build_vgg_1dcnn(self, w_init=None, n_layers=(4, 2, 1), poolsize=(2), n_filters_first=32, filter_size=(3), size_fc=1024, output=True):
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
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(self.imsize, self.n_colors)))
        # initialize counter to keep track of which weight to assign
        count = 0
        # add the rest of the hidden layers
        for idx, n_layer in enumerate(n_layers):
            for ilay in range(n_layer):
                self.model.add(Conv1D(n_filters_first*(2 ** idx),
                                      kernel_size=filter_size,
                                      input_shape=(self.imsize, self.n_colors),
                                      kernel_initializer=w_init[count],
                                      activation='relu'))
                # increment counter to the next weight initializer
                count += 1
            # create a network at the end with a max pooling
            self.model.add(MaxPooling1D(pool_size=poolsize))

        if output:
            self.buildoutput(size_fc)

    def build_vgg_2dcnn(self, w_init=None, n_layers=(4, 2, 1), poolsize=(2, 2), n_filters_first=32, filter_size=(3, 3), size_fc=1024, output=True):
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
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(
            self.imsize, self.imsize, self.n_colors)))
        # initialize counter to keep track of which weight to assign
        count = 0
        # add the rest of the hidden layers
        for idx, n_layer in enumerate(n_layers):
            for ilay in range(n_layer):
                self.model.add(Conv2D(n_filters_first*(2 ** idx),
                                      kernel_size=filter_size,
                                      input_shape=(
                                          self.imsize, self.imsize, self.n_colors),
                                      kernel_initializer=w_init[count],
                                      activation='linear'))
                self.model.add(LeakyReLU(alpha=0.1))
                # increment counter to the next weight initializer
                count += 1
            # create a network at the end with a max pooling
            self.model.add(MaxPooling2D(pool_size=poolsize))

        if output:
            self.buildoutput(size_fc)

    def build_vgg_3dcnn(self, w_init=None, n_layers=(4, 2, 1), poolsize=(2, 2, 2), n_filters_first=32, filter_size=(3, 3, 3), size_fc=1024, output=True):
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
        DEBUG = 0
        # check for weight initialization -> apply Glorotuniform
        if w_init is None:
            w_init = [keras.initializers.glorot_uniform()] * sum(n_layers)
        # set up input layer of CNN
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(
            self.imsize, self.imsize, self.imsize, self.n_colors)))
        # initialize counter to keep track of which weight to assign
        count = 0
        # add the rest of the hidden layers
        for idx, n_layer in enumerate(n_layers):
            for ilay in range(n_layer):
                self.model.add(Conv3D(n_filters_first*(2 ** idx),
                                      kernel_size=filter_size,
                                      # input_shape=(self.imsize, self.imsize, self.n_colors),
                                      kernel_initializer=w_init[count],
                                      activation='relu'))
                # increment counter to the next weight initializer
                count += 1
            # create a network at the end with a max pooling
            self.model.add(MaxPooling3D(pool_size=poolsize))

        if output:
            self.buildoutput(size_fc)

    def build_inception2dcnn(self, num_layers=10, n_filters_first=64, size_fc=1024):
        '''
        Build our own customized inception style 2d cnn for our data.

        Allows customization based on number of layers, layout, etc.
        '''
        # define the input image
        input_img = Input(shape=(self.imsize, self.imsize, self.n_colors))

        # first add the beginning convolutional layers
        conv_input_img = Conv2D(n_filters_first//2, 
                            kernel_size=(3,3),
                            strides=(2,2),
                            padding='valid',
                            activation='relu')(input_img)
        conv_input_img = Conv2D(n_filters_first//2, 
                            kernel_size=(3,3),
                            strides=(1,1),
                            padding='valid',
                            activation='relu')(conv_input_img)
        conv_input_img = Conv2D(n_filters_first, 
                            kernel_size=(3,3),
                            strides=(1,1),
                            padding='same',
                            activation='relu')(conv_input_img)
        conv_input_img = MaxPooling2D((3,3), 
                        strides=(2,2), 
                        padding='same')(conv_input_img)

        # add the inception modules
        for i in range(num_layers):
            if i == 0:
                numfilters = n_filters_first
            else:
                numfilters = n_filters_first // i

            # build the inception towers
            if i == 0:
                tower_0, tower_1, tower_2, tower_3 = self._build_inception_towers(conv_input_img, numfilters)
            else:
                output = MaxPooling2D((3,3), strides=(2,2), 
                        padding='same')(output)
                tower_0, tower_1, tower_2, tower_3 = self._build_inception_towers(output, numfilters)  
            # concatenate the layers and flatten
            output = keras.layers.concatenate([tower_0, tower_1, tower_2, tower_3], axis=1)

        # make sure layers are all flattened
        output = Flatten()(output)
        # output = AveragePooling1D(pool_size=4,
        #                           strides=1)(output)
        # output = AveragePooling2D(pool_size=(4,4),
        #                         strides=1)(output)
        # create the output layers that are generally fc - with some DROPOUT
        output = self._build_output(output, size_fc=size_fc)
        # create the model
        self.model = Model(inputs=input_img, outputs=output)
        return self.model

    def _build_inception_towers(self, input_img, n_filters_first=64):
        '''
        Utility function to build up the inception modules for an
        inception style network that operates at different scales (e.g.
        1x1, 3x3, 5x5 convolutions)
        '''

        # create the towers that occur during each layer of diff scale convolutions
        tower_0 = Conv2D(n_filters_first, kernel_size=(1,1),
                        padding='same',
                        activation='relu')(input_img)
        tower_1 = Conv2D(n_filters_first, kernel_size=(1,1), 
                        padding='same', 
                        activation='relu')(input_img)
        tower_1 = Conv2D(n_filters_first, kernel_size=(3,3), 
                        padding='same',
                        activation='relu')(tower_1)
        tower_2 = Conv2D(n_filters_first, kernel_size=(1,1), 
                        padding='same', 
                        activation='relu')(input_img)
        tower_2 = Conv2D(n_filters_first, kernel_size=(5,5), 
                        padding='same', 
                        activation='relu')(tower_2)
        tower_3 = MaxPooling2D((3,3), strides=(1,1), 
                        padding='same')(input_img)
        tower_3 = Conv2D(n_filters_first, kernel_size=(1,1), 
                        padding='same', 
                        activation='relu')(tower_3)

        return tower_0, tower_1, tower_2, tower_3