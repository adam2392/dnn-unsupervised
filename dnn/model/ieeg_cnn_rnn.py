############################ NUM PROCESSING FXNS ############################
import numpy as np
np.random.seed(1234)
### Necessary libraries for interfacing with Data
import scipy.io
from scipy.interpolate import griddata
from sklearn.preprocessing import scale

############################ UTILITY FUNCTIONS ############################
import time
from functools import reduce
import math as m

############################ ANN FUNCTIONS ############################
######### import DNN for training using GPUs #########
# from keras.utils.training_utils import multi_gpu_model

######### import DNN frameworks #########
import tensorflow as tf
import keras

from keras import Model

# import high level optimizers, models and layers
from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.layers import InputLayer

# for CNN
from keras.layers import Conv1D, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D
# for RNN
from keras.layers import LSTM
from keras.layers import Bidirectional
# for general NN behavior
from keras.layers import TimeDistributed, Dense, Dropout, Flatten
from keras.layers import Input, Concatenate, Permute, Reshape, Merge


# utility functionality for keras
# from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding

# preprocessing
# from keras.preprocessing.image import ImageDataGenerator
# from utils import augment_EEG, cart2sph, pol2cart
# from keras import backend as K


class IEEGdnn():
    def __init__(self, imsize =32, n_colors =3, num_classes = 2):
        '''
        Parameters:
        num_classes         (int) the number of classes in prediction space
        '''
        # initialize class elements
        self.imsize = imsize
        self.n_colors = n_colors
        self.num_classes = num_classes

        # start off with a relatively simple sequential model
        self.model = Sequential()

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
        DEBUG=0
        # check for weight initialization -> apply Glorotuniform
        if w_init is None:
            w_init = [keras.initializers.glorot_uniform()] * sum(n_layers)
        # set up input layer of CNN
        model = Sequential()
        model.add(InputLayer(input_shape=(self.imsize, self.imsize, self.n_colors)))
        # initialize counter to keep track of which weight to assign
        count=0
        # add the rest of the hidden layers
        for idx, n_layer in enumerate(n_layers):
            for ilay in range(n_layer):
                model.add(Conv2D(n_filters_first*(2 ** idx), 
                                 kernel_size=filter_size,
                                 input_shape=(self.imsize, self.imsize, self.n_colors),
                                 kernel_initializer=w_init[count], 
                                 activation='relu'))
                if DEBUG:
                    print(model.output_shape)
                    print(idx, " and ", ilay)
                # increment counter to the next weight initializer
                count+=1
            # create a network at the end with a max pooling
            model.add(MaxPooling2D(pool_size=poolsize))
        return model

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
        model = Sequential()
        model.add(InputLayer(input_shape=(self.imsize, self.imsize, self.imsize, self.n_colors)))
        # initialize counter to keep track of which weight to assign
        count=0
        # add the rest of the hidden layers
        for idx, n_layer in enumerate(n_layers):
            for ilay in range(n_layer):
                model.add(Conv3D(n_filters_first*(2 ** idx), 
                                 kernel_size=filter_size,
                                 # input_shape=(self.imsize, self.imsize, self.n_colors),
                                 kernel_initializer=w_init[count], 
                                 activation='relu'))
                if DEBUG:
                    print(model.output_shape)
                    print(idx, " and ", ilay)
                # increment counter to the next weight initializer
                count+=1
            # create a network at the end with a max pooling
            model.add(MaxPooling3D(pool_size=poolsize))
        return model

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
        self.model.add(Embedding(input_dim=input_dim, output_dim=embed_vector_dim, input_length=input_len))
        self.model.add(LSTM(size_mem))
        self.model.add(Dense(output_dim, activation='relu'))
        return self.model

    def build_same_cnn_lstm(self, num_timewins, size_mem= 128,size_fc=1024, dim=2, BIDIRECT=True, DROPOUT= False):
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
        w_init = None

        # create a single CNN
        if dim == 2:
            convnet = self._build_2dcnn(w_init=w_init, n_layers=(4,2,1), 
                    poolsize=(2,2), n_filters_first=32, filter_size=(3,3))
        elif dim==3:
            convnet = self._build_3dcnn(w_init=w_init, n_layers=(4,2,1), 
                poolsize=(2,2,2), n_filters_first=32, filter_size=(3,3,3))
        # flatten layer from single CNN (e.g. model.output_shape == (None, 64, 32, 32) -> (None, 65536))
        convnet.add(Flatten())
        cnn_output_shape = convnet.output_shape[1]

        # create sequential model to get this all before the LSTM
        model = Sequential()
        if dim == 2:
            model.add(TimeDistributed(convnet, input_shape=(num_timewins, self.imsize, self.imsize, self.n_colors)))
        elif dim==3:
            model.add(TimeDistributed(convnet, input_shape=(num_timewins, self.imsize, self.imsize, self.imsize, self.n_colors)))
        if BIDIRECT:
            model.add(Bidirectional(LSTM(units=size_mem, 
                            activation='relu', 
                            return_sequences=False)))
        else:
            model.add(LSTM(units=size_mem, 
                            activation='relu', 
                            return_sequences=False))
        output = self._build_seq_output(model, size_fc=size_fc)
        
        return output
        # model.add(Input(tensor=output))
        # return model

    def build_cnn_lstm(self, num_timewins, size_mem= 128, 
        size_fc=1024, dim=2, BIDIRECT=True, DROPOUT= False):
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
        w_init = None

        # initialize list of CNN that we want
        convnets = []
        # Build 7 parallel CNNs with shared weights
        for i in range(num_timewins):
            if dim == 2:
                convnet = self._build_2dcnn(w_init=w_init, n_layers=(4,2,1), 
                        poolsize=(2,2), n_filters_first=32, filter_size=(3,3))
            elif dim==3:
                convnet = self._build_3dcnn(w_init=w_init, n_layers=(4,2,1), 
                    poolsize=(2,2,2), n_filters_first=32, filter_size=(3,3,3))
            convnet.add(Flatten())
            # adds a flattened layer for the convnet (e.g. model.output_shape == (None, 64, 32, 32) -> (None, 65536))
            convnets.append(convnet)

        model = Sequential()
        # create a concatenated layer from all the parallel CNNs
        model.add(Merge(convnets, mode='concat'))

        # reshape the output layer to be #timewins x features 
        # (i.e. chans*rows*cols)
        num_cnn_features = convnets[0].output_shape[1]
        model.add(Reshape((num_timewins, num_cnn_features)))
        ########################## Build into LSTM now #################
        # Input to LSTM should have the shape as (batch size, seqlen/timesteps, inputdim/features)
        if BIDIRECT:
            model.add(Bidirectional(LSTM(units=size_mem, 
                            activation='relu', 
                            return_sequences=False)))
        else:
            # only get the last LSTM output
            model.add(LSTM(units=size_mem, 
                            activation='relu', 
                            return_sequences=False))

        model = self._build_seq_output(model, size_fc)
        return model

    def build_cnn_lstm_mix(self, num_timewins, size_mem= 128, size_fc = 1024, dim=2, BIDIRECT=True, DROPOUT= False):
        '''
        - NEED TO DETERMINE HOW TO FEED SEPARATE DATA INTO EACH OF THE CNN'S...
        CAN'T BE BUILT SEQUENTIALLY?
        - FIX ERRORS WRT LASAGNE VS KERAS
        '''
        w_init = None
        # initialize list of CNN that we want
        convnets = []
        # Build 7 parallel CNNs with shared weights
        for i in range(num_timewins):
            if dim == 2:
                convnet = self._build_2dcnn(w_init=w_init, n_layers=(4,2,1), 
                        poolsize=(2,2), n_filters_first=32, filter_size=(3,3))
            elif dim==3:
                convnet = self._build_3dcnn(w_init=w_init, n_layers=(4,2,1), 
                    poolsize=(2,2,2), n_filters_first=32, filter_size=(3,3,3))
            convnet.add(Flatten())
            # adds a flattened layer for the convnet (e.g. model.output_shape == (None, 64, 32, 32) -> (None, 65536))
            convnets.append(convnet)

        model = Sequential()
        # create a concatenated layer from all the parallel CNNs
        model.add(Merge(convnets, mode='concat'))
        # reshape the output layer to be #timewins x features 
        # (i.e. chans*rows*cols)
        num_cnn_features = convnets[0].output_shape[1]
        model.add(Reshape((num_timewins, num_cnn_features)))
        convpool = model.output

        ########################## Build separate output from 1d conv layer #################
        # this is input into the 1D conv layer | reshaped features x timewins
        reform_convpool = Permute((2,1))(convpool)
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
        finalmodel = keras.layers.concatenate([lstm, convout_1d])
        finalmodel = self._build_output(finalmodel, size_fc)

        finalmodel = Model(inputs=model.input, outputs=finalmodel)
        return finalmodel
        
    def _build_output(self, finalmodel, size_fc=1024, DROPOUT= False):
        '''
        Creates the final output layers of the sequential model: a fully connected layer
        followed by a final classification layer.

        Parameters:
        size_fc             (int) the size of the fully connected dense layer
        DROPOUT             (bool) True, of False on whether to use dropout or not

        Returns:
        model               the sequential model object with all layers added in LSTM style,
                            or the actual tensor
        '''
        # finalmodel = Sequential()
        # finalmodel.add(InputLayer(input_shape=model.output_shape))
        # if DROPOUT:
        #     finalmodel.add(Dropout(0.5))
        # finalmodel.add(Dense(size_fc, activation='relu'))
        # if DROPOUT:
        #     finalmodel.add(Dropout(0.5))
        # # final classification layer -> softmax for multiclass, 
        # finalmodel.add(Dense(num_classes, activation='softmax'))

        # finalmodel = Flatten()(finalmodel)
        output = Dense(size_fc, activation='relu')(finalmodel)
        if DROPOUT:
            output = Dropout(0.5)(output)
        output = Dense(self.num_classes, activation='softmax')(output)
        if DROPOUT:
            output = Dropout(0.5)(output)
        return output

    def _build_seq_output(self, finalmodel, size_fc=1024, DROPOUT= False):
        '''
        Creates the final output layers of the sequential model: a fully connected layer
        followed by a final classification layer.

        Parameters:
        size_fc             (int) the size of the fully connected dense layer
        DROPOUT             (bool) True, of False on whether to use dropout or not

        Returns:
        model               the sequential model object with all layers added in LSTM style,
                            or the actual tensor
        '''
        # finalmodel = Sequential()
        # finalmodel.add(InputLayer(input_shape=model.output_shape))
        # if DROPOUT:
        #     finalmodel.add(Dropout(0.5))
        # finalmodel.add(Dense(size_fc, activation='relu'))
        # if DROPOUT:
        #     finalmodel.add(Dropout(0.5))
        # # final classification layer -> softmax for multiclass, 
        # finalmodel.add(Dense(num_classes, activation='softmax'))
        if len(finalmodel.output_shape) > 2:
            finalmodel.add(Flatten())
        if DROPOUT:
            finalmodel.add(Dropout(0.5))
        finalmodel.add(Dense(size_fc, activation='relu'))
        if DROPOUT:
            finalmodel.add(Dropout(0.5))
        finalmodel.add(Dense(self.num_classes, activation='softmax'))
        return finalmodel

    def init_callbacks(self):
        callbacks = [LearningRateScheduler(poly_decay)]

        return callbacks
    '''
    Functions for completing and running the entire model
    '''
    def compile_model(self, model, loss='categorical_crossentropy', optimizer=None, metrics=['accuracy']):
        optimizer = keras.optimizers.Adam(lr=0.001, 
                                        beta_1=0.9, 
                                        beta_2=0.999,
                                        epsilon=1e-08,
                                        decay=0.0)
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        
        # store the final configuration of the model
        self.model_config = model.get_config()
        return model, model.get_config()

    def train(self, model, xtrain, ytrain, xtest, ytest,
        batch_size=32, epochs=10, AUGMENT=False):
        '''
        Main function to train the DNN model constructed:

        Things to Add:
        1. shuffling of minibatches
        2.  
        '''
        if AUGMENT:
            callbacks = self.init_callbacks()
            aug = ImageDataGenerator(width_shift_range=0.1,
                    height_shift_range=0.1, horizontal_flip=True,
                    fill_mode="nearest")
            HH = model.fit_generator(
                aug.flow(xtrain, ytrain, batch_size=batch_size * G), # adds augmentation to data using generator
                validation_data=(xtest, ytest),  
                steps_per_epoch=len(xtrain) // (batch_size * G),    #
                epochs=epochs,
                callbacks=callbacks, verbose=2)
        else:
            HH = model.fit(xtrain, ytrain, verbose=1, batch_size=batch_size, epochs=epochs)
        self.HH = HH
        return HH

    def eval(self, xtest, ytest, batch_size=32):
        self.score = self.model.evaluate(xtest, ytest, batch_size=batch_size)

        acc_train_history = self.score.history['acc']
        acc_test_history = self.score.history['val_acc']
        loss_train_history = self.score.history['loss']
        loss_test_history = self.score.history['val_loss']
