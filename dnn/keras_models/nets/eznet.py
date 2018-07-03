############################ NUM PROCESSING FXNS ############################
import numpy as np
############################ ANN FUNCTIONS ############################
######### import DNN frameworks #########
import tensorflow as tf
import keras

from dnn.keras_models.nets.base import BaseNet
import dnn.base.constants.model_constants as MODEL_CONSTANTS

from dnn.keras_models.nets.generic import tcn, vgg

from keras import Model
# import high level optimizers, models and layers
from keras.models import Sequential, Model
from keras.layers import InputLayer

# for CNN
from keras.layers import Conv2D, Conv1D
from keras.layers import MaxPooling1D, MaxPooling2D
# for general NN behavior
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Input, Concatenate, Permute, Reshape, SpatialDropout1D
from keras.layers import Activation
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
        self.tcn = None
        self.auxnet = None

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
        self.size_fc = 512

        # parameters for AuxNet
        numfilters = 24
        poolsize=((1,2))
        kernel_size=(1,2)
        dilation = (1,1)
        nb_stacks = 1
        n_layers = [4, 2, 1]

        vgg = self.build_vgg(n_layers,
                    poolsize,
                    numfilters,
                    kernel_size, 
                    nb_stacks)

        # parameters for TCN
        dilations = [1]
        numfilters = 24
        kernel_size = 2
        nb_stacks=1
        activation = 'relu'

        tcn = self.build_dilatedtcn(dilations, 
                            numfilters, kernel_size, 
                            nb_stacks, activation=activation)
        combinedx = self.combinenets(tcn, vgg)

        if output:
            combinedx = self.buildoutput(tcn, self.size_fc)

        # net = Model(inputs=[self.aux_input_layer, self.tcn_input_layer], outputs=combinedx)
        net = Model(inputs=self.tcn_input_layer, outputs=combinedx)
        self.net = net
        return net

    def buildoutput(self, model, size_fc):
        model = self._build_output(model, size_fc=size_fc)
        return model 

    def build_dilatedtcn(self, dilations, 
                    numfilters,
                    kernel_size, 
                    nb_stacks,
                    activation='norm_relu',
                    use_skip_connections=False):
        # if self.w_init is None:
        # check for weight initialization -> apply Glorotuniform
        w_init = [keras.initializers.glorot_uniform()] * nb_stacks*len(dilations)

        # define starting layers
        input_layer = Input(name='input_layer', shape=(self.width_imsize, self.n_colors))
        x = input_layer
        self.tcn_input_layer = input_layer

        x = Conv1D(numfilters, kernel_size, 
                        kernel_initializer=w_init[0],
                        padding='causal', 
                        name='initial_conv')(x)

        # keep track of output of each output
        skip_connections = []
        for s in range(nb_stacks):
            for i in dilations:
                kernel_init = keras.initializers.he_normal()
                x, skip_out = tcn.TCN.residual_block(x, s, i, activation, 
                                            nb_filters=numfilters, 
                                            kernel_size=kernel_size,
                                            kernel_init=kernel_init)
                skip_connections.append(skip_out)

        # should we use skip_connections?
        if use_skip_connections:
            x = keras.layers.add(skip_connections)

        x = Flatten()(x)
        self.tcn = x
        return x

    def build_vgg(self, n_layers,
                    poolsize,
                    numfilters,
                    kernel_size, 
                    nb_stacks):
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
        # if self.w_init is None:
        # w_init = [keras.initializers.glorot_uniform()]
        
        # define starting layers
        input_layer = Input(name='aux_input_layer', 
                            shape=(self.length_imsize, self.width_imsize, self.n_colors))
        x = input_layer
        self.aux_input_layer = input_layer

        # initialize counter to keep track of which weight to assign
        count = 0
        # add the rest of the hidden layers
        vgg_helper = vgg.VGG(self.length_imsize, self.width_imsize, self.n_colors)
        for s in range(nb_stacks):
            for idx, n_layer in enumerate(n_layers):
                for ilay in range(n_layer):
                    # kernel_init = keras.initializers.glorot_uniform()
                    kernel_init = keras.initializers.he_normal()
                    x = vgg_helper.residualblock(x, ilay, idx,
                                            numfilters, 
                                            kernel_size,
                                            kernel_init)
                    # increment counter to the next weight initializer
                    count += 1
                # create a network at the end with a max pooling
                x = MaxPooling2D(pool_size=poolsize)(x)

        x = Flatten()(x)

        self.auxnet = x
        return x

    def combinenets(self, tcn, auxnet):
        # concatenate the two outputs
        x = keras.layers.concatenate([tcn, auxnet])
        return x

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

