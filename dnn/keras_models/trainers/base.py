import numpy as np 
from dnn.base.constants.config import Config, OutputConfig
from dnn.base.utils.log_error import initialize_logger
import dnn.base.constants.model_constants as constants

from dnn.keras_models.metrics.classifier import BinaryClassifierMetric
from dnn.base.utils.data_structures_utils import NumpyEncoder
import os
import json
import pickle
import io
try:
    to_unicode = unicode
except NameError:
    to_unicode = str
class TrainMetrics(object):
    # metrics
    loss_queue = []
    recall_queue = []
    precision_queue = []
    accuracy_queue = []
    fp_queue = []


class TestMetrics(object):
    # metrics
    loss_queue = []
    recall_queue = []
    precision_queue = []
    accuracy_queue = []
    fp_queue = []


class BaseTrainer(object):
    metric_comp = BinaryClassifierMetric()
    model = None 
    def __init__(self, model, config=None):
        self.config = config or Config()
        self.logger = initialize_logger(
            self.__class__.__name__,
            self.config.out.FOLDER_LOGS)
        self.model = model

    def configure(self):
        msg = "Base trainer configure method is not implemented."
        raise NotImplementedError(msg)

    def train(self):
        msg = "Base trainer train method is not implemented."
        raise NotImplementedError(msg)

    def _summarize(self, outputs, labels, loss, regularize=False):
        pass

    def savemetricsoutput(self, modelname):
        metricsfilepath = os.path.join(self.outputdatadir, modelname+ "_metrics.json")
        auc = self.metrichistory.aucs
        fpr = self.metrichistory.fpr
        tpr = self.metrichistory.tpr 
        thresholds = self.metrichistory.thresholds
        metricdata = {
            'auc': auc,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
        }
        self._writejsonfile(metricdata, metricsfilepath)

    def saveoutput(self, modelname):
        modeljson_filepath = os.path.join(self.outputdatadir, modelname + "_model.json")
        history_filepath = os.path.join(
            self.outputdatadir,  modelname + '_history' + '.pkl')
        finalweights_filepath = os.path.join(
            self.outputdatadir, modelname + '_final_weights' + '.h5')
        self._saveoutput(modeljson_filepath, history_filepath, finalweights_filepath)

    def _saveoutput(self, modeljson_filepath, history_filepath, finalweights_filepath):
        # save model
        if not os.path.exists(modeljson_filepath):
            # serialize model to JSON
            model_json = self.model.to_json()
            self._writejsonfile(model_json, modeljson_filepath)
            # with open(modeljson_filepath, "w") as json_file:
            #     json_file.write(model_json)
            print("Saved model to disk")

        # save history
        with open(history_filepath, 'wb') as file_pi:
            pickle.dump(self.HH.history, file_pi)
        print("saved history file!")
        
        # save final weights
        self.model.save(finalweights_filepath)
        print("saved final weights file!")

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

        self.metadata = metadata
        return self.metadata

        