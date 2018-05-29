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
from keras.layers import Conv1D
# for RNN
from keras.layers import LSTM
from keras.layers import Bidirectional
# for general NN behavior
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Input, Concatenate, Permute, Reshape, Merge

from keras.layers import concatenate
from keras.optimizers import Adam

# utility functionality for keras
# from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers import TimeDistributed, Dense, Dropout, Flatten

import json
import pickle


class iEEGSeq(BaseNet):
    def __init__(self, name, num_classes=2, num_timewins=5, DROPOUT=True, BIDIRECT=False, FREEZE=True):
        '''
        Parameters:
        num_classes         (int) the number of classes in prediction space
        '''
        _availmodels = ['SAME', 'CNNLSTM', 'MIX']
        if name not in _availmodels:
            raise AttributeError('Name needs to be one of the following:'
                                 'SAMECNNLSTM CNNLSTM MIX')

        # initialize class elements
        self.name = name
        self.num_classes = num_classes              # number of classes to predict
        # number of windows to look at per sequence
        self.num_timewins = num_timewins
        self.DROPOUT = DROPOUT
        self.BIDIRECT = BIDIRECT
        self.FREEZE = FREEZE

        # start off with a relatively simple sequential model
        self.model = Sequential()

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
    def loadmodel(self, modelfile, weightsfile):
        # load json and create model
        json_file = open(modelfile, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        # fixed_cnn_model = ieegdnn.load_model(weightsfile, freeze=True)
        fixed_cnn_model = keras.models.model_from_json(loaded_model_json)
        fixed_cnn_model.load_weights(weightsfile)

        # remove the last 2 dense FC layers and freeze it
        fixed_cnn_model.pop()
        fixed_cnn_model.pop()

        return fixed_cnn_model

    def buildmodel(self, convnet):
        size_mem = 128
        num_timewins = self.num_timewins
        self.input_shape = convnet.input_shape

        # self.model = convnet
        if self.name == 'SAME':
            self._build_same_cnn_lstm(
                convnet, num_timewins=num_timewins, size_mem=size_mem, BIDIRECT=self.BIDIRECT)
        elif self.name == 'CNNLSTM':
            self._build_cnn_lstm(
                convnet, num_timewins=num_timewins, size_mem=size_mem, BIDIRECT=self.BIDIRECT)
        elif self.name == 'MIX':
            self._build_cnn_lstm_mix(
                convnet, num_timewins=num_timewins, size_mem=size_mem, BIDIRECT=self.BIDIRECT)
        elif self.name == 'LOAD':
            self._appendlstm(
                convnet, num_timewins=num_timewins, size_mem=size_mem)

    def buildoutput(self):
        size_fc = 1024  # size of the FC layer

        if self.name == 'SAME' or self.name == 'CNNLSTM':
            self._build_seq_output(size_fc=size_fc)
            # self.model = Model(inputs=self.model.input, outputs = self.model.output)
        elif self.name == 'MIX':
            self._build_output(self.auxmodel, size_fc)
            # self.model = Model(inputs=self.model.input, outputs=self.output)
        elif self.name == 'LOAD':
            self.model = self._build_seq_output(size_fc=size_fc)
            # self.model = Model(inputs=self.model.input, outputs = self.model.output)
        return self.model

    def _build_same_cnn_lstm(self, convnet, num_timewins, size_mem=128, BIDIRECT=True):
        '''
        Creates a CNN network with shared weights, with a LSTM layer to 
        integrate time from sequences of images 

        Parameters:
        num_timewins        (int) the number of time windows in this snip
        size_mem            (int) the number of memory units to use 
        DROPOUT             (bool) True, of False on whether to use dropout or not

        Returns:
        model               the sequential model object with all layers added in LSTM style
        '''
        if self.FREEZE:
            # convnet.compile()
            convnet.trainable = False
            # convnet.compile()

        # flatten layer from single CNN (e.g. model.output_shape == (None, 64, 32, 32) -> (None, 65536))
        convnet.add(Flatten())
        # store the output shape (total number of features)
        cnn_output_shape = convnet.output_shape[1]
        cnn_input_shape = tuple(list(convnet.input_shape)[1:])

        # self.model = convnet
        # create sequential model to get this all before the LSTM
        self.model.add(TimeDistributed(
            convnet, input_shape=(num_timewins,)+cnn_input_shape))
        if BIDIRECT:
            self.model.add(Bidirectional(LSTM(units=size_mem,
                                              activation='relu',
                                              return_sequences=False)))
        else:
            self.model.add(LSTM(units=size_mem,
                                activation='relu',
                                return_sequences=False))

    def _build_cnn_lstm(self, convnet, num_timewins, size_mem=128, BIDIRECT=True):
        '''
        Creates a CNN network with shared weights, with a LSTM layer to 
        integrate time from sequences of images 

        Parameters:
        num_timewins        (int) the number of time windows in this snip
        size_mem            (int) the number of memory units to use 
        DROPOUT             (bool) True, of False on whether to use dropout or not

        Returns:
        model               the sequential model object with all layers added in LSTM style
        '''
        # initialize list of CNN that we want
        convnets = []
        buffweights = convnet.weights
        convnet.add(Flatten())

        # Build 7 parallel CNNs with shared weights
        for i in range(num_timewins):
            # adds a flattened layer for the convnet (e.g. model.output_shape == (None, 64, 32, 32) -> (None, 65536))
            convnets.append(convnet)

        # create a concatenated layer from all the parallel CNNs
        self.model.add(Merge(convnets, mode='concat'))

        # reshape the output layer to be #timewins x features
        # (i.e. chans*rows*cols)
        num_cnn_features = convnets[0].output_shape[1]
        self.model.add(Reshape((num_timewins, num_cnn_features)))
        ########################## Build into LSTM now #################
        # Input to LSTM should have the shape as (batch size, seqlen/timesteps, inputdim/features)
        if BIDIRECT:
            self.model.add(Bidirectional(LSTM(units=size_mem,
                                              activation='relu',
                                              return_sequences=False)))
        else:
            # only get the last LSTM output
            self.model.add(LSTM(units=size_mem,
                                activation='relu',
                                return_sequences=False))

    def _build_cnn_lstm_mix(self, convnet, num_timewins, size_mem=128, BIDIRECT=True):
        '''
        - NEED TO DETERMINE HOW TO FEED SEPARATE DATA INTO EACH OF THE CNN'S...
        CAN'T BE BUILT SEQUENTIALLY?
        - FIX ERRORS WRT LASAGNE VS KERAS
        '''
       # initialize list of CNN that we want
        convnets = []
        buffweights = convnet.weights
        convnet.add(Flatten())

        cnn_input_shape = tuple(list(convnet.input_shape)[1:])
        # Build 7 parallel CNNs with shared weights
        for i in range(num_timewins):
            # adds a flattened layer for the convnet (e.g. model.output_shape == (None, 64, 32, 32) -> (None, 65536))
            convnets.append(convnet)

        if self.FREEZE:
            for net in convnets:
                net.trainable = False

        # create a concatenated layer from all the parallel CNNs
        # self.model.add(InputLayer(input_shape=(num_timewins,)+cnn_input_shape))
        # create a concatenated layer from all the parallel CNNs
        self.model.add(Merge(convnets, mode='concat'))
        # self.model.add(Concatenate(convnets))
        # self.model.add(convnets)

        # reshape the output layer to be #timewins x features
        # (i.e. chans*rows*cols)
        num_cnn_features = convnets[0].output_shape[1]
        self.model.add(Reshape((num_timewins, num_cnn_features)))
        convpool = self.model.output

        ########################## Build separate output from 1d conv layer #################
        # this is input into the 1D conv layer | reshaped features x timewins
        reform_convpool = Permute((2, 1))(convpool)
        # input to 1D convlayer should be in (batch_size, num_input_channels, input_length)
        convout_1d = Conv1D(filters=64, kernel_size=3)(reform_convpool)
        convout_1d = Flatten()(convout_1d)

        ########################## Build into LSTM now #################
        # Input to LSTM should have the shape as (batch size, seqlen/timesteps, inputdim/features)
        # only get the last LSTM output
        if BIDIRECT:
            lstm = Bidirectional(LSTM(units=size_mem,
                                      activation='relu',
                                      return_sequences=False))(convpool)
        else:
            lstm = LSTM(units=size_mem,
                        activation='relu',
                        return_sequences=False)(convpool)

        # Merge 1D-Conv and LSTM outputs -> feed into the final fc / classify layers
        self.auxmodel = keras.layers.concatenate([lstm, convout_1d])

    def _build_lstm(self, input_dim, embed_vector_dim, input_len, output_dim, size_mem):
        '''
        Creates a LSTM network in some default style. It requires self
        to initialize a sequential model first.

        Parameters:
        input_dim           (int) size of vocabulary, or size of possible total prediction 
                            (e.g. if you have 500 words in your dataset, this is 500)
        embed_vector_dim    (int) the output dimensions / dense embedding dimension
        input_len           (int) the length of the input sequences when constant. 
                            This argument is required if you are going to connect Flatten 
                            then Dense layers upstream (without it, the shape of the dense
                            outputs cannot be computed).
        output_dim          (int) the output dimensions of the network (e.g. binary classification 
                            would be 0, or 1, so output_dim=1)
        size_mem            (int) the number of memory units to use 

        Returns:
        model               the sequential model object with all layers added in LSTM style
        '''
        self.model.add(Embedding(input_dim=input_dim,
                                 output_dim=embed_vector_dim, input_length=input_len))
        self.model.add(LSTM(size_mem))
        self.model.add(Dense(output_dim, activation='relu'))
        return self.model

    def _appendlstm(self, fixed_model, num_timewins, size_mem, BIDIRECT):
        # create sequential model to get this all before the LSTM
        self.model.add(TimeDistributed(
            fixed_model, input_shape=(num_timewins,)+self.input_shape))
        if BIDIRECT:
            self.model.add(Bidirectional(LSTM(units=size_mem,
                                              activation='relu',
                                              return_sequences=False)))
        else:
            self.model.add(LSTM(units=size_mem,
                                activation='relu',
                                return_sequences=False))
