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

from keras import Model
# import high level optimizers, models and layers
from keras.models import Sequential, Model
from keras.layers import InputLayer

# for CNN
from keras.layers import Conv1D, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D
# for general NN behavior
from keras.layers import Dense, Dropout, Flatten
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

    def buildmodel(self):
        w_init = None                       # weight initialization
        n_layers = (4,2,1)                  # number of convolutions per layer
        numfilters = 32                     # number of filters in first layer of each new layer
        poolsize = ((2,)*self.modeldim)      # pool size   
        filter_size = ((3,)*self.modeldim)   # filter size

        self.w_init = w_init
        self.n_layers = n_layers
        self.numfilters = numfilters
        self.poolsize = poolsize
        self.filter_size = filter_size

        if self.modeldim == 1:
            self._build_1dcnn(w_init=w_init, 
                n_layers=n_layers, 
                poolsize=poolsize, 
                n_filters_first=numfilters, 
                filter_size=filter_size)
        elif self.modeldim == 2:
            self._build_2dcnn(w_init=w_init, 
                n_layers=n_layers, 
                poolsize=poolsize, 
                n_filters_first=numfilters, 
                filter_size=filter_size)
        elif self.modeldim == 3:
            self._build_3dcnn(w_init=w_init, 
                n_layers=n_layers, 
                poolsize=poolsize, 
                n_filters_first=numfilters, 
                filter_size=filter_size)
        else:
            raise ValueError('Model dimension besides (1,2,3) not implemented!')


    def buildoutput(self):
        size_fc = 1024
        self._build_seq_output(size_fc=size_fc)

    def _build_1dcnn(self, w_init= None, n_layers = (4,2,1), poolsize = (2), n_filters_first=32, filter_size=(3)):    
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
        count=0
        # add the rest of the hidden layers
        for idx, n_layer in enumerate(n_layers):
            for ilay in range(n_layer):
                self.model.add(Conv1D(n_filters_first*(2 ** idx), 
                                 kernel_size=filter_size,
                                 input_shape=(self.imsize, self.n_colors),
                                 kernel_initializer=w_init[count], 
                                 activation='relu'))
                # increment counter to the next weight initializer
                count+=1
            # create a network at the end with a max pooling
            self.model.add(MaxPooling2D(pool_size=poolsize))

    def _build_2dcnn(self, w_init= None, n_layers = (4,2,1), poolsize = (2,2), n_filters_first = 32, filter_size=(3,3)):    
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
        self.model.add(InputLayer(input_shape=(self.imsize, self.imsize, self.n_colors)))
        # initialize counter to keep track of which weight to assign
        count=0
        # add the rest of the hidden layers
        for idx, n_layer in enumerate(n_layers):
            for ilay in range(n_layer):
                self.model.add(Conv2D(n_filters_first*(2 ** idx), 
                                 kernel_size=filter_size,
                                 input_shape=(self.imsize, self.imsize, self.n_colors),
                                 kernel_initializer=w_init[count], 
                                 activation='relu'))
                # increment counter to the next weight initializer
                count+=1
            # create a network at the end with a max pooling
            self.model.add(MaxPooling2D(pool_size=poolsize))

    def _build_3dcnn(self, w_init= None, n_layers = (4,2,1), poolsize = (2,2,2), n_filters_first = 32, filter_size=(3,3,3)):    
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
        DEBUG=0
        # check for weight initialization -> apply Glorotuniform
        if w_init is None:
            w_init = [keras.initializers.glorot_uniform()] * sum(n_layers)
        # set up input layer of CNN
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(self.imsize, self.imsize, self.imsize, self.n_colors)))
        # initialize counter to keep track of which weight to assign
        count=0
        # add the rest of the hidden layers
        for idx, n_layer in enumerate(n_layers):
            for ilay in range(n_layer):
                self.model.add(Conv3D(n_filters_first*(2 ** idx), 
                                 kernel_size=filter_size,
                                 # input_shape=(self.imsize, self.imsize, self.n_colors),
                                 kernel_initializer=w_init[count], 
                                 activation='relu'))
                # increment counter to the next weight initializer
                count+=1
            # create a network at the end with a max pooling
            self.model.add(MaxPooling3D(pool_size=poolsize))