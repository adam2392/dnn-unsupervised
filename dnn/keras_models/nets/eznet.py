############################ NUM PROCESSING FXNS ############################
import numpy as np
############################ ANN FUNCTIONS ############################
######### import DNN frameworks #########
import tensorflow as tf
import keras

from dnn.keras_models.nets.base import BaseNet
import dnn.base.constants.model_constants as MODEL_CONSTANTS

from keras import Model
# import high level optimizers, models and layers
from keras.models import Sequential, Model
from keras.layers import InputLayer

# for CNN
from keras.layers import Conv2D, Conv1D
from keras.layers import MaxPooling1D, MaxPooling2D
# for general NN behavior
from keras.layers import Dense, Dropout, Flatten, ReLU, BatchNormalization
from keras.layers import Input, Concatenate, Permute, Reshape

import pprint

class EZNet(BaseNet):
    def __init__(self, length_imsize=30, 
                        width_imsize=60,
                        n_colors=1, 
                        num_classes=2, 
                        config=None):
        '''
        Parameters:
        num_classes         (int) the number of classes in prediction space
        '''
        super(EZNet, self).__init__(config=config)

        # initialize class elements
        self.length_imsize = length_imsize
        self.width_imsize = width_imsize
        self.n_colors = n_colors
        self.num_classes = num_classes

        # initialize model constants
        self.DROPOUT = MODEL_CONSTANTS.DROPOUT

        # start off with a relatively simple sequential model
        self.net = None
        # initialize the two input networks - vector and auxiliary image
        self.vecnet = Sequential()
        self.auxnet = Sequential()

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
            'filteringsize': self.kernel_size
        }
        pprint.pprint(summary)

    def buildmodel(self, output=True):
        # weight initialization
        self.w_init = None  
        # number of convolutions per layer
        self.n_layers = (4, 2, 1)
        self.numfilters = 24                     # number of filters in first layer of each new layer
        self.poolsize = ((2,)*self.netdim)      # pool size
        self.kernel_size = ((1,2)*self.netdim)   # filter size
        self.size_fc = 1024
        self.dilation = (1,1) 

        # parameters for TCN
        dilations = [1,2,4,6]
        numfilters = 24
        kernel_size = 3
        padding_type='causal'
        nb_stacks=4

        self._build_vgg(dilations, numfilters, 
                        kernel_size, 
                        padding_type, nb_stacks)

        ''' INCEPTION MODEL PARAMS '''
        # self.num_layers=10, 
        # self.n_filters_first=64
        # self._build_inception()
        if output:
            self.buildoutput(self.size_fc)

    def buildoutput(self, size_fc):
        self._build_seq_output(size_fc=size_fc)

    def _buildauxnet(self):
        self._build_vgg()

    def _build_dilatedtcn(self, dilations, numfilters, kernel_size, 
                            padding_type, nb_stacks):
        # check for weight initialization -> apply Glorotuniform
        w_init = [keras.initializers.glorot_uniform()] * sum(nb_stacks)
        self.vecnet.add(InputLayer(name='input_layer', input_shape=(self.width_imsize, self.n_colors)))


    def _build_vec(self):
        # check for weight initialization -> apply Glorotuniform
        if self.w_init is None:
            self.w_init = [keras.initializers.glorot_uniform()] * sum(self.n_layers)
        self.vecnet.add(InputLayer(input_shape=(self.width_imsize, self.n_colors)))
        
        # initialize counter to keep track of which weight to assign
        count = 0
        # add the rest of the hidden layers
        for idx, n_layer in enumerate(self.n_layers):
            for ilay in range(n_layer):
                kernel_init = self.w_init[count]
                self._add_vgg_layer(idx, kernel_init)
                # increment counter to the next weight initializer
                count += 1
            # create a network at the end with a max pooling
            self.auxnet.add(MaxPooling2D(pool_size=self.poolsize))

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

    def _add_vgg_layer(self, idx, kernel_init):
        self.auxnet.add(Conv2D(self.numfilters*(2 ** idx),
                          kernel_size=self.kernel_size,
                          input_shape=(
                          self.imsize, self.imsize, self.n_colors),
                          dilation_rate=self.dilation,
                          kernel_initializer=kernel_init,
                          activation='linear'))
        # self.auxnet.add(LeakyReLU(alpha=0.1))
        self.auxnet.add(BatchNormalization(axis=-1, momentum=0.99, 
            epsilon=0.001, center=True, scale=True, 
            beta_initializer='zeros', gamma_initializer='ones', 
            moving_mean_initializer='zeros', moving_variance_initializer='ones', 
            beta_regularizer=None, gamma_regularizer=None, 
            beta_constraint=None, gamma_constraint=None))
        self.auxnet.add(ReLU())

    def _build_vgg(self):
        '''
        Creates a Convolutional Neural network in VGG-16 style. It requires self
        to initialize a sequential model first.

        Parameters:
        w_init              (list) of all the weights (#layers * #nodes_in_layers)
        n_layers            (tuple) of number of nodes in each layer
        poolsize            (tuple) for max pooling poolsize along each dimension 
                            (e.g. 2D is (2,2) for pooling over 2 pixels in x and y direction)
        n_filters_first     (int) number of filters in the first layer
        kernel_size         (int/tuple/list) the kernel/filter size of the width/height of
                            2D convolution window

        Returns:
        model               the sequential model object with all layers added in CNN style
        '''
        # check for weight initialization -> apply Glorotuniform
        if self.w_init is None:
            self.w_init = [keras.initializers.glorot_uniform()] * sum(self.n_layers)
        self.auxnet.add(InputLayer(input_shape=(self.length_imsize, self.width_imsize, self.n_colors)))
        # initialize counter to keep track of which weight to assign
        count = 0
        # add the rest of the hidden layers
        for idx, n_layer in enumerate(self.n_layers):
            for ilay in range(n_layer):
                kernel_init = self.w_init[count]
                self._add_vgg_layer(idx, kernel_init)
                # increment counter to the next weight initializer
                count += 1
            # create a network at the end with a max pooling
            self.auxnet.add(MaxPooling2D(pool_size=self.poolsize))

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


if __name__ == '__main__':
    # define model
    model_params = {
        'numclasses': 2,
        'imsize': 32,
        'n_colors':4,
    }
    model = iEEGCNN(**model_params) 
    model.buildmodel(output=True)

    print(model)

