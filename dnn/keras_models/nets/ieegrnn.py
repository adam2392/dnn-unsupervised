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

from keras.optimizers import Adam
# for CNN
from keras.layers import RNN, GRU, LSTM
# for general NN behavior
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Input, Concatenate, Permute, Reshape, Merge

# utility functionality for keras
# from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
import pprint


class iEEGRNN(BaseNet):
    def __init__(self, num_times=500, num_layers=2, hidden_size=500, num_classes=2, DROPOUT=True):
        '''
        Parameters:
        num_classes         (int) the number of classes in prediction space
        '''
        # initialize class elements
        self.num_times = num_times
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.DROPOUT = DROPOUT

        # start off with a relatively simple sequential model
        self.model = Sequential()

    def summaryinfo(self):
        summary = {
            'num_times': self.num_times,
            'numclasses': self.num_classes,
            'numlayers': self.num_layers,
            'DROPOUT': self.DROPOUT,
            'w_init': self.w_init,
            'hidden_size': self.hidden_size,
        }
        pprint.pprint(summary)

    def buildmodel(self, celltype='lstm', output=True):
        # weight initialization
        size_fc = 1024

        # build up the deep RNN
        self._build_deeprnn(celltype=celltype)
        # build the final fully connected layers for the output.
        if output:
            self.buildoutput(size_fc)

    def buildoutput(self, size_fc):
        self._build_seq_output(size_fc=size_fc)

    def _build_deeprnn(self, celltype='lstm'):
        '''
        Creates a deep recurrent style neural network, that can
        either be rnn, lstm, or a gru. It requires self
        to initialize a sequential model first.

        Parameters:
        w_init              (list) of all the weights (#layers * #nodes_in_layers)
        celltype            (string) is the cell type we use for our deep 
                            recurrent neural network

        Returns:
        model               the sequential model object with all layers added in CNN style
        '''
        if celltype not in ['rnn', 'lstm', 'gru']:
            raise ValueError('This celltype is not supported! Only the rnn, lstm and gru cells are supported')

        # set up input layer of RNN
        self.model = Sequential()
        self.model.add(Embedding(self.num_times, self.hidden_size, input_length=self.num_times))
        
        for idx in range(self.num_layers - 1):
            if celltype is 'lstm':
                self.model.add(LSTM(units=self.hidden_size, 
                            activation='relu', 
                            return_sequences=True))
            elif celltype is 'gru':
                self.model.add(GRU(units=self.hidden_size, 
                            activation='relu', 
                            return_sequences=True))
            elif celltype is 'rnn':
                self.model.add(RNN(units=self.hidden_size, 
                            activation='relu', 
                            return_sequences=True))
        
        # add the final block which does not return sequences, but just 
        # the last single output
        if celltype is 'lstm':
            self.model.add(LSTM(units=self.hidden_size, 
                            activation='relu', 
                            return_sequences=False))
        elif celltype is 'gru':
            self.model.add(GRU(units=self.hidden_size, 
                            activation='relu', 
                            return_sequences=False))
        elif celltype is 'rnn':
            self.model.add(RNN(units=self.hidden_size, 
                            activation='relu', 
                            return_sequences=False)) 
