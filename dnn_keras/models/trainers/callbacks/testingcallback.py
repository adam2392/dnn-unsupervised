from keras.callbacks import Callback
import numpy as np
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import confusion_matrix, classification_report

from dnn_keras.models.metrics.classifier import BinaryClassifierMetric

class MetricsCallback(Callback):
    def on_train_begin(self, logs={}):
        self.metrics = BinaryClassifierMetric()
        self.aucs = []
        self.roc_metrics = []
        self.fpr = []
        self.tpr = []
        self.thresholds = []
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, logs={}):
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        # access the validatian data
        x = self.validation_data[0]
        y = self.validation_data[1]

        # compute loss, and accuracy of model
        loss, acc = self.model.evaluate(x, y, verbose=0)
        predicted_probs = self.model.predict(x)

        # compute roc_auc scores using the predicted probabilties
        self.metrics.compute_roc(ytrue, predicted_probs)
        # extract the receiver operating curve statistics
        fpr, tpr, thresholds = metrics.roc 
        self.fpr.append(fpr)
        self.tpr.append(tpr)
        self.thresholds.append(thresholds)
        self.aucs.append(roc_auc_score(y, predicted_probs))

        # compute the predicted classes
        predicted = self.model.predict_classes(x)
        ytrue = np.argmax(y, axis=1)

        metrics.compute_metrics(ytrue, predicted)
        print('Testing loss: {}, acc: {}'.format(loss, acc))
        print('Mean accuracy score: {}'.format(self.metrics.accuracy))
        print('Recall: {}'.format(self.metrics.recall))
        print('Precision: {}'.format(self.metrics.precision))
        print("FPR: {}".format(self.metrics.fp))
        print('\n clasification report:\n',
              classification_report(ytrue, predicted))
        print('\n confusion matrix:\n', confusion_matrix(ytrue, predicted))
