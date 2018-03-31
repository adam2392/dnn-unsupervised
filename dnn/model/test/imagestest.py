from .basetest import BaseTest
import numpy as np
import os
from sklearn.model_selection import train_test_split
# utilitiy libs
import ntpath
import json
import pickle

import keras
import sklearn.utils

# metrics for postprocessing of the results
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, \
    recall_score, classification_report, \
    f1_score, roc_auc_score

import pprint
import warnings


class TestCNN(BaseTest):
    def __init__(self, model, X_test=None, y_test=None):
        self.dnnmodel = dnnmodel
        self.X_test = X_test
        self.y_test = y_test

        if X_test is None or y_test is None:
            warnings.warn(
                'X_test or y_test is not passed in. Make sure to load in the testing data.')

    def saveoutput(self, outputdatadir):
        modeljsonfile = os.path.join(outputdatadir, modelname + "_model.json")
        historyfile = os.path.join(
            outputdatadir,  modelname + '_history' + '.pkl')
        finalweightsfile = os.path.join(
            outputdatadir, modelname + '_final_weights' + '.h5')

        # save model
        if not os.path.exists(modeljsonfile):
            # serialize model to JSON
            model_json = self.dnnmodel.to_json()
            with open(modeljsonfile, "w") as json_file:
                json_file.write(model_json)
            print("Saved model to disk")

        # save history
        with open(historyfile, 'wb') as file_pi:
            pickle.dump(self.HH.history, file_pi)

        # save final weights
        self.dnnmodel.save(finalweightsfile)

    def configure(self, tempdatadir):
        # initialize loss function, SGD optimizer and metrics
        loss = 'binary_crossentropy'
        optimizer = Adam(lr=1e-5,
                         beta_1=0.9,
                         beta_2=0.99,
                         epsilon=1e-08,
                         decay=0.0)
        metrics = ['accuracy']
        self.modelconfig = self.dnnmodel.compile(loss=loss,
                                                 optimizer=optimizer,
                                                 metrics=metrics)

        tempfilepath = os.path.join(
            tempdatadir, "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5")

        # callbacks availabble
        checkpoint = ModelCheckpoint(tempfilepath,
                                     monitor='val_acc',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='max')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                      patience=10, min_lr=1e-8)
        testcheck = TestCallback()
        self.callbacks = [checkpoint, testcheck]

    def loadtestdata(self):
        '''     LOAD TESTING DATA      '''
        for idx, datafile in enumerate(self.testfilepaths):
            imagedata = np.load(datafile)
            image_tensor = imagedata['image_tensor']
            metadata = imagedata['metadata'].item()

            if idx == 0:
                image_tensors = image_tensor
                ylabels = metadata['ylabels']
            else:
                image_tensors = np.append(image_tensors, image_tensor, axis=0)
                ylabels = np.append(ylabels, metadata['ylabels'], axis=0)
        # load the ylabeled data 1 in 0th position is 0, 1 in 1st position is 1
        invert_y = 1 - ylabels
        ylabels = np.concatenate((invert_y, ylabels), axis=1)
        # format the image tensor correctly
        image_tensors = self._formatdata(image_tensors)
        self.X_test = image_tensors
        self.y_test = ylabels

    def seq_testdata(self):
    '''
    Function to feed in the test data and get the outputs 
    sequentially.
    
    '''
        pass
