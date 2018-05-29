import numpy as np 
import json
import pandas as pd 

from dnn_pytorch.models.metrics.base import BaseMetric

import sklearn
from sklearn import metrics

class BinaryClassifierMetric(BaseMetric):
    def __init__(self, config=None):
        super(BinaryClassifierMetric, self).__init__(config=config)

    def compute_metrics(self, y_true, y_pred, sample_weight=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.sample_weight = sample_weight

        self.recall = self._recall(y_true, y_pred, sample_weight)
        self.precision = self._precision(y_true, y_pred, sample_weight)
        self.accuracy = self._accuracy(y_true, y_pred, sample_weight)
        self.tn, self.fp, self.fn, self.tp = self._confusion(y_true, y_pred, sample_weight)

        self.logger.info("Computed scores: recall, precision and confusion matrix!")

    def compute_doa(self, y_true, y_pred):
        pass

    def compute_roc(self, y_true, y_pred_probs):
        self.roc = metrics.roc_curve(y_true, y_pred_probs, 
                                    pos_label=1, 
                                    sample_weight=None, 
                                    drop_intermediate=True)

    def compute_class_weights(self, y_true):
        self.class_weights = sklearn.utils.compute_class_weight('balanced',
                                          np.unique(
                                              y_true).astype(int),
                                          np.argmax(y_true, axis=1))
    def fpr(self, fp, tn, totaltime):
        return fp/(fp+tn)

    def _recall(self, y_true, y_pred, sample_weight):
        '''
        AKA Sensitivity
        '''
        recall = metrics.recall_score(y_true, y_pred, 
                                        pos_label=1,
                                        sample_weight=sample_weight)
        return recall
    def _precision(self, y_true, y_pred, sample_weight):
        precision = metrics.precision_score(y_true, y_pred, 
                                        pos_label=1,
                                        sample_weight=sample_weight)
        return precision
    def _accuracy(self, y_true, y_pred, sample_weight):
        _accuracy = metrics.accuracy_score(y_true, y_pred, 
                                        normalize=True,
                                        sample_weight=sample_weight)
        return accuracy
    def _fp(self,y_true, y_pred, sample_weight):
        _, fp, _, _ = metrics.confusion_matrix(y_true, y_pred).ravel()
        return fp

    def _confusion(self, y_true, y_pred, sample_weight):
        tp, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        return tp, fp, fn, tp