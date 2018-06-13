import numpy as np
import os
import keras
from keras.utils.training_utils import multi_gpu_model

# Custom Built libraries
import dnn_keras
from dnn_keras.base.implementation.base import BaseHPC
from dnn_keras.models.nets.cnn import iEEGCNN
from dnn_keras.io.readerimgdataset import ReaderImgDataset 
import dnn_keras.base.constants.model_constants as constants
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def load_data(traindir, testdir, data_procedure='loo', testpat=None, training_pats=None):
    '''
    If LOO training, then we have to trim these into 
    their separate filelists
    '''
    # initialize reader to get the training/testing data
    reader = ReaderImgDataset()
    reader.loadbydir(traindir, testdir, procedure=data_procedure, testname=testpat)
    reader.loadfiles(mode=constants.TRAIN)
    reader.loadfiles(mode=constants.TEST)
    
    # create the dataset objects
    train_dataset = reader.train_dataset
    test_dataset = reader.test_dataset

    return train_dataset, test_dataset

def createmodel(num_classes, imsize, n_colors):
    # define model
    model_params = {
        'num_classes': 2,
        'imsize': 64,
        'n_colors':4,
    }
    model = iEEGCNN(**model_params) 
    model.buildmodel(output=True)

    return model

def trainmodel(model, num_epochs, batch_size, train_dataset, test_dataset, 
                    testpatdir, expname, device=None):
    if device is None:
        devices = get_available_gpus()
    if len(devices) > 0:
        print("Let's use {} GPUs!".format(len(devices)))
        # make the model parallel
        model = multi_gpu_model(model, gpus=len(devices))

    trainer = CNNTrainer(model=model.net, num_epochs=num_epochs, 
                        batch_size=batch_size,
                        testpatdir=testpatdir)
    trainer.composedatasets(train_dataset, test_dataset)
    trainer.configure()
    # Train the model
    trainer.train()
    print(model.net)
    print("Training on {} ".format(device))
    print("Training object: {}".format(trainer))
    return trainer

def testmodel(trainer, modelname):
    trainer.saveoutput(modelname=modelname)
    trainer.savemetricsoutput(modelname=modelname)
    return trainer

