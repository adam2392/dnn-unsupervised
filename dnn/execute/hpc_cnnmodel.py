import numpy as np
import os
import keras
from keras.utils.training_utils import multi_gpu_model

# Custom Built libraries
import dnn
from dnn.execute.hpc_basemodel import BaseHPC

import dnn.base.constants.model_constants as constants
from dnn.keras_models.nets.cnn import iEEGCNN
from dnn.keras_models.trainers.cnn import CNNTrainer
from dnn.io.readerimgdataset import ReaderImgDataset 

class MarccHPC(BaseHPC):
    '''
    An implementation specifcally for our MARCC HPC runner that
    impelements the basehpc functions.
    '''
    @staticmethod
    def load_test_data(traindir, testdir, 
        data_procedure='loo', 
        testpat=None, 
        training_pats=None):
        '''
        If LOO training, then we have to trim these into 
        their separate filelists
        '''
        # initialize reader to get the training/testing data
        reader = ReaderImgDataset()
        reader.loadbydir(traindir, testdir, procedure=data_procedure, testname=testpat)
        reader.loadfiles(mode=constants.TEST)

        # create the dataset objects
        test_dataset = reader.test_dataset
        return test_dataset

    @staticmethod
    def load_data(traindir, testdir, 
        data_procedure='loo', 
        testpat=None, 
        training_pats=None):
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

    @staticmethod
    def load_data_files(trainfilepaths, testfilepaths, 
        data_procedure='loo', testpat=None, training_pats=None):
        '''
        If LOO training, then we have to trim these into 
        their separate filelists
        '''
        # initialize reader to get the training/testing data
        reader = ReaderImgDataset()
        reader.loadfiles(trainfilepaths, mode=constants.TRAIN)
        reader.loadfiles(testfilepaths, mode=constants.TEST)

        # create the dataset objects
        train_dataset = reader.train_dataset
        test_dataset = reader.test_dataset
        return train_dataset, test_dataset

    @staticmethod
    def createmodel(num_classes,
                    imsize, n_colors,
                    weightsfilepath=None, modelfilepath=None):
        # define model parameters
        model_params = {
            'num_classes': num_classes,
            'imsize': imsize,
            'n_colors': n_colors,
        }
        model = iEEGCNN(**model_params)

        # load in old model
        if modelfilepath is not None and weightsfilepath is not None:
            print("loading in model and weights from {} and {}".format(modelfilepath, weightsfilepath))
            model.loadmodel_file(modelfilepath, weightsfilepath)
        else:
            model.buildmodel(output=True)
        return model

    @staticmethod
    def trainmodel(model, num_epochs, batch_size, 
                    train_dataset, test_dataset, 
                    outputdir, expname, use_dir_generator=False, device=None):
        if device is None:
            devices = super(MarccHPC, MarccHPC).get_available_gpus()
        if len(devices) > 1:
            print("Let's use {} GPUs!".format(len(devices)))
            # make the model parallel
            model = multi_gpu_model(model, gpus=len(devices))

        trainer = CNNTrainer(model=model, 
                            num_epochs=num_epochs, 
                            batch_size=batch_size,
                            outputdir=outputdir)
        if not use_dir_generator:
            trainer.composedatasets(train_dataset, test_dataset)
        trainer.configure()
        print("Training on {} ".format(device))
        print("Training object: {}".format(trainer))
        # Train the model
        trainer.train()
        return trainer

    @staticmethod
    def testmodel(trainer, modelname):
        trainer.saveoutput(modelname=modelname)
        trainer.savemetricsoutput(modelname=modelname)
        return trainer

