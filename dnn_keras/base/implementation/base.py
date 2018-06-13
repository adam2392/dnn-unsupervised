import numpy as np
import os
from tensorflow.python.client import device_lib

class BaseHPC(object):
    @staticmethod
    def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

    @staticmethod
    def load_data(traindir, testdir, data_procedure='loo', testpat=None, training_pats=None):
        '''
        If LOO training, then we have to trim these into 
        their separate filelists
        '''
        raise NotImplementedError("Need to implement load data that returns train and test dataset objects.")
        return train_dataset, test_dataset

    @staticmethod
    def createmodel(num_classes, imsize, n_colors):
        # define model
        raise NotImplementedError("Need to implement createmodel that \
                    returns the model to use.")
        return model

    def trainmodel(model, num_epochs, batch_size, train_dataset, test_dataset, 
                        testpatdir, expname, device=None):
        raise NotImplementedError("Need to implement trainmodel that \
                    train the model we pass with the datasets.")
        return trainer

    def testmodel(trainer, modelname):
        raise NotImplementedError("Need to implement testmodel that \
                    test the trained model and output some end metrics.")
        return trainer

