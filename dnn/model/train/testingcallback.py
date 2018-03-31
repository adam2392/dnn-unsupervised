from keras.callbacks import Callback
import numpy as np

class TestCallback(Callback):
    def __init__(self):
        # self.test_data = test_data
        self.aucs = []

    def on_epoch_end(self, epoch, logs={}):
        # x, y = self.test_data
        # x = self.model.validation_data[0]
        # y = self.model.validation_data[1]
        x = self.validation_data[0]
        y = self.validation_data[1]

        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

        predicted = self.model.predict(x)
        self.aucs.append(roc_auc_score(y, predicted))

        predicted = self.model.predict_classes(x)
        ytrue = np.argmax(y, axis=1)
        print('Mean accuracy score: ', accuracy_score(ytrue, predicted))
        print('F1 score:', f1_score(ytrue, predicted))
        print('Recall:', recall_score(ytrue, predicted))
        print('Precision:', precision_score(ytrue, predicted))
        print('\n clasification report:\n',
              classification_report(ytrue, predicted))
        print('\n confusion matrix:\n', confusion_matrix(ytrue, predicted))
