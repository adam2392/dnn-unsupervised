############################ NUM PROCESSING FXNS ############################
import numpy as np
############################ ANN FUNCTIONS ############################
######### import DNN frameworks #########
import tensorflow as tf
import keras

from dnn.keras_models.nets.base import BaseNet
import dnn.base.constants.model_constants as MODEL_CONSTANTS

from dnn.keras_models.nets.generic import inception
from dnn.keras_models.nets.generic import tcn, vgg

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
        self.length_imsize = imsize
        self.width_imsize = imsize
        self.n_colors = n_colors
        self.num_classes = num_classes

        self.netdim = modeldim

        # initialize model constants
        self.DROPOUT = MODEL_CONSTANTS.DROPOUT

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
        self.size_fc = 512
        # number of convolutions per layer
        numfilters = 32
        poolsize=((2,2))
        kernel_size=(3,3)
        dilation = (1,1)
        nb_stacks = 1
        n_layers = [4, 2, 1]

        vgg = self.build_vgg(n_layers,
                    poolsize,
                    numfilters,
                    kernel_size, 
                    nb_stacks)

        ''' INCEPTION MODEL PARAMS '''
        # num_layers=10, 
        # n_filters_first=64
        # inception = self._build_inception()
        if output:
            model_output = self.buildoutput(vgg, self.size_fc)

        self.net = Model(inputs=self.input_layer,
                        outputs=model_output)

    def buildoutput(self, model, size_fc):
        model = self._build_output(model, size_fc=size_fc)
        return model 

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
        input_layer = Input(name='input_layer', 
                            shape=(self.length_imsize, self.width_imsize, self.n_colors))
        x = input_layer
        self.input_layer = input_layer

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
        self.net = x
        return x

    def _build_inception(self):
        '''
        Build our own customized inception style 2d cnn for our data.

        Allows customization based on number of layers, layout, etc.
        '''
        # define the input image
        input_img = Input(name='input_layer', shape=(self.imsize, self.imsize, self.n_colors))
        self.input_layer = input_img

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
                output = inception.Inception().residualblock(conv_input_img, i, numfilters)
            else:
                output = MaxPooling2D((3,3), strides=(2,2), 
                        padding='same')(output)
                output = inception.Inception().residualblock(output, i, numfilters)  

        # make sure layers are all flattened
        output = Flatten()(output)

        return output


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

