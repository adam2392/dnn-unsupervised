from abc import ABCMeta, abstractmethod
from keras.layers import TimeDistributed, Dense, Dropout, Flatten

from dnn_keras.base.constants.config import Config
from dnn_keras.base.utils.log_error import initialize_logger

'''
Base class neural network for our models that we build.
'''


class BaseNet(metaclass=ABCMeta):
    requiredAttributes = ['DROPOUT', 'num_classes']
    
    def __init__(self, config=None):
        if config is not -1:
            self.config = config or Config()
            self.logger = initialize_logger(
                self.__class__.__name__,
                self.config.out.FOLDER_LOGS)

    # @property
    # def DROPOUT(self):
    #     msg = "All models need to say whether or not they have DROPOUT (True, or False)."
    #     raise NotImplementedError(msg)

    # @property
    # def num_classes(self):
    #     msg = "All models need to have the number of classes to predict."
    #     raise NotImplementedError(msg)

    def buildoutput(self):
        msg = "Base build model method is not implemented."
        raise NotImplementedError(msg)

    def buildmodel(self):
        msg = "Base build model method is not implemented."
        raise NotImplementedError(msg)

    def _build_output(self, finalmodel, size_fc):
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
        # finalmodel = Flatten()(finalmodel)
        self.output = Dense(size_fc, activation='relu')(finalmodel)
        if self.DROPOUT:
            self.output = Dropout(0.5)(self.output)
        self.output = Dense(size_fc/2, activation='relu')(self.output)
        if self.DROPOUT:
            self.output = Dropout(0.5)(self.output)
        self.output = Dense(
            self.num_classes, activation='softmax')(self.output)
        if self.DROPOUT:
            self.output = Dropout(0.5)(self.output)
        return self.output

    def _build_seq_output(self, size_fc=1024):
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
        if len(self.model.output_shape) > 2:
            self.model.add(Flatten())
        if self.DROPOUT:
            self.model.add(Dropout(0.5))
        self.model.add(Dense(size_fc, activation='relu'))
        if self.DROPOUT:
            self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))
