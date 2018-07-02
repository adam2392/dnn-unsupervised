from keras.callbacks import Callback
import numpy as np
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import confusion_matrix, classification_report

from dnn.keras_models.metrics.classifier import BinaryClassifierMetric

class MetricsCallback(Callback):
    def __init__(self):
        super(MetricsCallback, self).__init__()
        
    def on_train_begin(self, logs={}):
        self.metrics = BinaryClassifierMetric()
        self.aucs = []
        self.roc_metrics = []
        self.fpr = []
        self.tpr = []
        self.thresholds = []
 
    # def on_train_end(self, logs={}):
    #     return
 
    # def on_epoch_begin(self, epoch, logs={}):
    #     return
 
    # def on_batch_begin(self, batch, logs={}):
    #     return
 
    # def on_batch_end(self, batch, logs={}):
    #     return

    def on_epoch_end(self, epoch, logs={}):
        # access the validatian data
        x = self.validation_data[0]
        aux_x = x[0]
        xvec = x[1]

        if epoch < 5:
            print(len(x))
            print(aux_x.shape)
            print(xvec.shape)

        y = self.validation_data[1]
        ytrue = np.argmax(y, axis=1)

        # compute loss, and accuracy of model
        loss, acc = self.model.evaluate({'aux_input_layer': aux_x,
                                        'input_layer': xvec}, y, verbose=0)
        predicted_probs = self.model.predict(x)
        predicted_probs_positive = predicted_probs[:,1]
        # compute the predicted classes
        predicted = self.model.predict_classes(x)
            
        # compute roc_auc scores using the predicted probabilties
        self.metrics.compute_roc(ytrue, predicted_probs_positive)
        # extract the receiver operating curve statistics
        fpr, tpr, thresholds = self.metrics.roc 
        self.fpr.append(fpr)
        self.tpr.append(tpr)
        self.thresholds.append(thresholds)

        # compute the AUC score
        self.aucs.append(roc_auc_score(ytrue, predicted_probs_positive))

        self.metrics.compute_metrics(ytrue, predicted)
        print('Testing loss: {}, acc: {}'.format(loss, acc))
        print('Mean accuracy score: {}'.format(self.metrics.accuracy))
        print('Recall: {}'.format(self.metrics.recall))
        print('Precision: {}'.format(self.metrics.precision))
        print("FPR: {}".format(self.metrics.fp))
        print('\n clasification report:\n',
              classification_report(ytrue, predicted))
        print('\n confusion matrix:\n', confusion_matrix(ytrue, predicted))
