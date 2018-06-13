from abc import ABCMeta, abstractmethod
from keras.layers import TimeDistributed, Dense, Dropout, Flatten

from dnn.base.utils.data_structures_utils import NumpyEncoder
from dnn.base.constants.config import Config
from dnn.base.utils.log_error import initialize_logger

import sklearn
import json
import pickle
import pprint
'''
Base class neural network for our models that we build.
'''

class BaseNet(metaclass=ABCMeta):
    num_classes = None
    model = None
    output = None 
    DROPOUT = False 
    AUGMENT = False
    class_weight = []
    learning_rate = None

    def __init__(self, config=None):
        if config is not -1:
            self.config = config or Config()
            self.logger = initialize_logger(
                self.__class__.__name__,
                self.config.out.FOLDER_LOGS)

    def buildoutput(self):
        msg = "Base build model method is not implemented."
        raise NotImplementedError(msg)

    def buildmodel(self):
        msg = "Base build model method is not implemented."
        raise NotImplementedError(msg)

    def saveoutput(self):
        msg = "Base save model method is not implemented.\
        Add model, history, and weights file paths!"
        raise NotImplementedError(msg)

    def summaryinfo(self):
        summary = {
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'dropout': self.DROPOUT,
            'epochs': self.NUM_EPOCHS,
            'augment': self.AUGMENT,
            'class_weight': self.class_weight,
        }
        pprint.pprint(summary)

    def _writejsonfile(self, metadata, metafilename):
        with io.open(metafilename, 'w', encoding='utf8') as outfile:
            str_ = json.dumps(metadata,
                              indent=4, sort_keys=True, cls=NumpyEncoder,
                              separators=(',', ': '), ensure_ascii=False)
            outfile.write(to_unicode(str_))

    def _loadjsonfile(self, metafilename):
        if not metafilename.endswith('.json'):
            metafilename += '.json'

        try:
            with open(metafilename, mode='r', encoding='utf8') as f:
                metadata = json.load(f)
            metadata = json.loads(metadata)
        except Exception as e:
            print(e)
            print("can't open metafile: {}".format(metafilename))
            with io.open(metafilename, encoding='utf-8', mode='r') as fp:
                json_str = fp.read() #json.loads(
            metadata = json.loads(json_str)

        return metadata

    def _saveoutput(self, modeljson_filepath,
            history_filepath, finalweights_filepath):
        # save model
        if not os.path.exists(modeljson_filepath):
            # serialize model to JSON
            model_json = self.net.to_json()
            with open(modeljsonfile, "w", encoding="utf8") as json_file:
                json_file.write(model_json)
            print("Saved model to disk")

        # save history
        with open(historyfile, 'wb') as file_pi:
            pickle.dump(self.HH.history, file_pi)

        # save final weights
        self.net.save(finalweights_filepath)

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
        self.output = Dense(self.num_classes, activation='softmax')(self.output)
        if self.DROPOUT:
            self.output = Dropout(0.5)(self.output)
        return self.output

    def _build_seq_output(self, size_fc):
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
        if len(self.net.output_shape) > 2:
            self.net.add(Flatten())
        if self.DROPOUT:
            self.net.add(Dropout(0.5))
        self.net.add(Dense(size_fc, activation='relu'))
        if self.DROPOUT:
            self.net.add(Dropout(0.5))
        self.net.add(Dense(size_fc//2, activation='relu'))
        if self.DROPOUT:
            self.net.add(Dropout(0.5))
        self.net.add(Dense(self.num_classes, activation='softmax'))
