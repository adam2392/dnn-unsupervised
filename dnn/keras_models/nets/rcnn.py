############################ NUM PROCESSING FXNS ############################
import numpy as np
############################ ANN FUNCTIONS ############################

from dnn.keras_models.nets.base import BaseNet
import dnn.base.constants.model_constants as MODEL_CONSTANTS

import tensorflow as tf
import keras
from keras import Model
# import high level optimizers, models and layers
from keras.models import Sequential, Model
from keras.layers import InputLayer

# for CNN
from keras.layers import Conv2D
from keras.layers import MaxPooling1D, MaxPooling2D, MaxPooling3D
from keras.layers import AveragePooling1D, AveragePooling2D
# for general NN behavior
from keras.layers import Dense, Dropout, Flatten, LeakyReLU
from keras.layers import Input, Concatenate, Permute, Reshape, Add

# for RNN
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Conv1D
from keras.layers import TimeDistributed, Dense, Dropout, Flatten

import pprint

class RCNN(BaseNet):
    def __init__(self, name, seqlen=10, imsize=32, 
                n_colors=3, num_classes=2, 
                modeldim=2, config=None):
        '''
        Parameters:
        num_classes         (int) the number of classes in prediction space
        '''
        super(RCNN, self).__init__(config=config)
        _availmodels = ['SAME', 'CNNLSTM', 'MIX', 'LOAD']
        if name not in _availmodels:
            raise AttributeError('Name needs to be one of the following:'
                                 'SAMECNNLSTM CNNLSTM MIX')
        self.name = name 

        # initialize class elements
        self.seqlen = seqlen
        self.imsize = imsize
        self.n_colors = n_colors
        self.num_classes = num_classes

        self.netdim = modeldim

        # initialize model constants
        self.DROPOUT = MODEL_CONSTANTS.DROPOUT
        self.BIDIRECT = True
        self.FREEZE = False

        # start off with a relatively simple sequential model
        self.net = Sequential()

    def loadmodel_file(self, modelfile, weightsfile):
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
        self.convnet = fixed_cnn_model

    def loadmodel_net(self, convnet):
        self.convnet = convnet

    def buildoutput(self, size_fc):
        self._build_seq_output(size_fc=size_fc)

    def buildmodel(self, output=True):
        size_mem = 128
        self.size_fc = 1024
        self.size_mem = size_mem
        num_timewins = self.seqlen
        self.input_shape = self.convnet.input_shape

        if self.name == 'SAME':
            self._build_same_cnn_lstm(num_timewins=num_timewins, size_mem=size_mem, BIDIRECT=self.BIDIRECT)
        elif self.name == 'CNNLSTM':
            self._build_cnn_lstm(num_timewins=num_timewins, size_mem=size_mem, BIDIRECT=self.BIDIRECT)
        elif self.name == 'MIX':
            self._build_cnn_lstm_mix(num_timewins=num_timewins, size_mem=size_mem, BIDIRECT=self.BIDIRECT)
        elif self.name == 'LOAD':
            self._appendlstm(num_timewins=num_timewins, size_mem=size_mem, BIDIRECT=self.BIDIRECT)

        if output:
            self.buildoutput(self.size_fc)
            # self._build_output(self.net, self.size_fc)
            # # create sequential model to get this all before the LSTM
            # input_shape = (num_timewins,)+self.convnet.input_shape[1:]
            # inp = Input(shape=input_shape)
            # self.net = Model(inputs=inp, outputs=self.output)

    def _build_same_cnn_lstm(self, num_timewins, size_mem=128, BIDIRECT=True):
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
            # self.convnet.compile()
            self.convnet.trainable = False
            # self.convnet.compile()

        # flatten layer from single CNN (e.g. model.output_shape == (None, 64, 32, 32) -> (None, 65536))
        self.convnet.add(Flatten())
        # store the output shape (total number of features)
        cnn_output_shape = self.convnet.output_shape[1]
        cnn_input_shape = tuple(list(self.convnet.input_shape)[1:])

        # self.net = self.convnet
        # create sequential model to get this all before the LSTM
        self.net.add(TimeDistributed(
            self.convnet, input_shape=(num_timewins,)+cnn_input_shape))
        if BIDIRECT:
            self.net.add(Bidirectional(LSTM(units=size_mem,
                                              activation='relu',
                                              return_sequences=False)))
        else:
            self.net.add(LSTM(units=size_mem,
                                activation='relu',
                                return_sequences=False))

    def _build_cnn_lstm(self, num_timewins, size_mem=128, BIDIRECT=True):
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
        print("Building cnn lstm")
        # initialize list of CNN that we want
        convnets = []
        buffweights = self.convnet.weights
        self.convnet.add(Flatten())

        # get output of a convnet output shape
        num_cnn_features = self.convnet.output_shape[1]

        # Build 7 parallel CNNs with shared weights
        for i in range(num_timewins):
            # adds a flattened layer for the self.convnet (e.g. model.output_shape == (None, 64, 32, 32) -> (None, 65536))
            convnets.append(self.convnet.output)

        inp = self.convnet.input 
        out = Concatenate()(convnets)
        conv_models = Model(inputs=inp, outputs=out)

        # create a concatenated layer from all the parallel CNNs
        self.net.add(conv_models)

        # reshape the output layer to be #timewins x features
        # (i.e. chans*rows*cols)
        self.net.add(Reshape((num_timewins, num_cnn_features)))
        ########################## Build into LSTM now #################
        # Input to LSTM should have the shape as (batch size, seqlen/timesteps, inputdim/features)
        if BIDIRECT:
            self.net.add(Bidirectional(LSTM(units=size_mem,
                                              activation='relu',
                                              return_sequences=False)))
        else:
            # only get the last LSTM output
            self.net.add(LSTM(units=size_mem,
                                activation='relu',
                                return_sequences=False))

    def _build_cnn_lstm_mix(self, num_timewins, size_mem=128, BIDIRECT=True):
        '''
        - NEED TO DETERMINE HOW TO FEED SEPARATE DATA INTO EACH OF THE CNN'S...
        CAN'T BE BUILT SEQUENTIALLY?
        - FIX ERRORS WRT LASAGNE VS KERAS
        '''
       # initialize list of CNN that we want
        convnets = []
        buffweights = self.convnet.weights
        self.convnet.add(Flatten())

        # get output of a convnet output shape
        num_cnn_features = self.convnet.output_shape[1]

        # Build 7 parallel CNNs with shared weights
        for i in range(num_timewins):
            # adds a flattened layer for the self.convnet (e.g. model.output_shape == (None, 64, 32, 32) -> (None, 65536))
            convnets.append(self.convnet.output)

        if self.FREEZE:
            for net in convnets:
                net.trainable = False

        # create a concatenated layer from all the parallel CNNs
        inp = self.convnet.input 
        out = Concatenate()(convnets)
        conv_models = Model(inputs=inp, outputs=out)

        # create a concatenated layer from all the parallel CNNs
        self.net.add(conv_models)

        # reshape the output layer to be #timewins x features
        # (i.e. chans*rows*cols)
        self.net.add(Reshape((num_timewins, num_cnn_features), name="merge_conv_lstm"))
        convpool = self.net.output

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

        # out = Concatenate()([lstm, convout_1d])                
        # print(out.shape)
        # merged_model = Model(inputs=inp, outputs=out)
        # Merge 1D-Conv and LSTM outputs -> feed into the final fc / classify layers
        # self.net.add(merged_model)
        # self.net.add(Add()([lstm, convout_1d]))

        self.net = keras.layers.concatenate([lstm, convout_1d], axis=-1)

    def _appendlstm(self, num_timewins, size_mem, BIDIRECT):
        # create sequential model to get this all before the LSTM
        input_shape = (num_timewins,)+self.convnet.input_shape[1:]

        # flatten out the convnet
        self.convnet.add(Flatten())

        # time distribute it
        self.net.add(TimeDistributed(self.convnet, input_shape=input_shape))
        if BIDIRECT:
            self.net.add(Bidirectional(LSTM(units=size_mem,
                                              activation='relu',
                                              return_sequences=False)))
        else:
            self.net.add(LSTM(units=size_mem,
                                activation='relu',
                                return_sequences=False))

