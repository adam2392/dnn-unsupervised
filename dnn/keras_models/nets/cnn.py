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
from keras.layers import Conv2D
from keras.layers import MaxPooling1D, MaxPooling2D, MaxPooling3D
from keras.layers import AveragePooling1D, AveragePooling2D
# for general NN behavior
from keras.layers import Dense, Dropout, Flatten, ReLU, BatchNormalization
from keras.layers import Input, Concatenate, Permute, Reshape

import pprint

class iEEGCNN(BaseNet):
    def __init__(self, imsize=32, n_colors=3, num_classes=2, 
                modeldim=2, config=None):
        '''
        Parameters:
        num_classes         (int) the number of classes in prediction space
        '''
        super(iEEGCNN, self).__init__(config=config)

        # initialize class elements
        self.imsize = imsize
        self.n_colors = n_colors
        self.num_classes = num_classes

        self.netdim = modeldim

        # initialize model constants
        self.DROPOUT = MODEL_CONSTANTS.DROPOUT

        # start off with a relatively simple sequential model
        self.net = Sequential()

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

    def loadmodel_file(self, modelfile, weightsfile):
        # load json and create model
        json_file = open(modelfile, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        # load model and then initialize with these weights
        fixed_cnn_model = keras.models.model_from_json(loaded_model_json)
        fixed_cnn_model.load_weights(weightsfile)

        # remove the last 2 dense FC layers and freeze it
        self.net = fixed_cnn_model

    def buildmodel(self, output=True):
        # weight initialization
        self.w_init = None  
        # number of convolutions per layer
        self.n_layers = (4, 2, 1)
        self.numfilters = 32                     # number of filters in first layer of each new layer
        self.poolsize = ((2,)*self.netdim)      # pool size
        self.kernel_size = ((2,)*self.netdim)   # filter size
        self.size_fc = 1024
        self.dilation = (1,1) 
        self._build_vgg()

        ''' INCEPTION MODEL PARAMS '''
        # self.num_layers=10, 
        # self.n_filters_first=64
        # self._build_inception()
        if output:
            self.buildoutput(self.size_fc)

    def buildoutput(self, size_fc):
        self._build_seq_output(size_fc=size_fc)

    def _add_vgg_layer(self, idx, kernel_init):
        self.net.add(Conv2D(self.numfilters*(2 ** idx),
                                      kernel_size=self.kernel_size,
                                      input_shape=(
                                      self.imsize, self.imsize, self.n_colors),
                                      dilation_rate=self.dilation,
                                      kernel_initializer=kernel_init,
                                      activation='linear'))
        # self.net.add(LeakyReLU(alpha=0.1))
        self.net.add(BatchNormalization(axis=-1, momentum=0.99, 
            epsilon=0.001, center=True, scale=True, 
            beta_initializer='zeros', gamma_initializer='ones', 
            moving_mean_initializer='zeros', moving_variance_initializer='ones', 
            beta_regularizer=None, gamma_regularizer=None, 
            beta_constraint=None, gamma_constraint=None))
        self.net.add(ReLU())

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
        # set up input layer of CNN
        self.net = Sequential()
        # check for weight initialization -> apply Glorotuniform
        if self.w_init is None:
            self.w_init = [keras.initializers.glorot_uniform()] * sum(self.n_layers)
        self.net.add(InputLayer(input_shape=(
            self.imsize, self.imsize, self.n_colors)))
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
            self.net.add(MaxPooling2D(pool_size=self.poolsize))

    def _build_inception(self):
        '''
        Build our own customized inception style 2d cnn for our data.

        Allows customization based on number of layers, layout, etc.
        '''
        # define the input image
        input_img = Input(shape=(self.imsize, self.imsize, self.n_colors))

        # first add the beginning convolutional layers
        conv_input_img = Conv2D(self.n_filters_first//2, 
                            kernel_size=(3,3),
                            strides=(2,2),
                            padding='valid',
                            activation='relu')(input_img)
        conv_input_img = Conv2D(self.n_filters_first//2, 
                            kernel_size=(3,3),
                            strides=(1,1),
                            padding='valid',
                            activation='relu')(conv_input_img)
        conv_input_img = Conv2D(self.n_filters_first, 
                            kernel_size=(3,3),
                            strides=(1,1),
                            padding='same',
                            activation='relu')(conv_input_img)
        conv_input_img = MaxPooling2D((3,3), 
                        strides=(2,2), 
                        padding='same')(conv_input_img)

        # add the inception modules
        for i in range(self.num_layers):
            if i == 0:
                numfilters = self.n_filters_first
            else:
                numfilters = self.n_filters_first // i

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
        output = self._build_output(output, size_fc=self.size_fc)
        # create the model
        self.net = Model(inputs=input_img, outputs=output)
        return self.net

    def _build_inception_towers(self, input_img):
        '''
        Utility function to build up the inception modules for an
        inception style network that operates at different scales (e.g.
        1x1, 3x3, 5x5 convolutions)
        '''

        # create the towers that occur during each layer of diff scale convolutions
        tower_0 = Conv2D(self.n_filters_first, kernel_size=(1,1),
                        padding='same',
                        activation='relu')(input_img)
        tower_1 = Conv2D(self.n_filters_first, kernel_size=(1,1), 
                        padding='same', 
                        activation='relu')(input_img)
        tower_1 = Conv2D(self.n_filters_first, kernel_size=(3,3), 
                        padding='same',
                        activation='relu')(tower_1)
        tower_2 = Conv2D(self.n_filters_first, kernel_size=(1,1), 
                        padding='same', 
                        activation='relu')(input_img)
        tower_2 = Conv2D(self.n_filters_first, kernel_size=(5,5), 
                        padding='same', 
                        activation='relu')(tower_2)
        tower_3 = MaxPooling2D((3,3), strides=(1,1), 
                        padding='same')(input_img)
        tower_3 = Conv2D(self.n_filters_first, kernel_size=(1,1), 
                        padding='same', 
                        activation='relu')(tower_3)

        return tower_0, tower_1, tower_2, tower_3

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

