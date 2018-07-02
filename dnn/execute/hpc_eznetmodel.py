import numpy as np
import os
import keras
from keras.utils.training_utils import multi_gpu_model

# Custom Built libraries
import dnn
from dnn.execute.hpc_basemodel import BaseHPC
import dnn.base.constants.model_constants as constants

from dnn.keras_models.nets.eznet import EZNet
from dnn.keras_models.trainers.eznet import EZNetTrainer
from dnn.io.readerauxdataset import ReaderEZNetDataset

class MarccHPC(BaseHPC):
    '''
    An implementation specifcally for our MARCC HPC runner that
    impelements the basehpc functions.
    '''
    @staticmethod
    def load_data(traindir, testdir):
        '''
        If LOO training, then we have to trim these into 
        their separate filelists
        '''
        # initialize reader to get the training/testing data
        reader = ReaderEZNetDataset()
        reader.loadbydir(traindir, testdir)
        reader.loadfiles(mode=constants.TRAIN)
        reader.loadfiles(mode=constants.TEST)

        # create the dataset objects
        train_dataset = reader.train_dataset
        test_dataset = reader.test_dataset

        print(reader.test_dataset.X_aux.shape)
        print(reader.test_dataset.X_chan.shape)
        print(reader.test_dataset.ylabels.shape)
        return train_dataset, test_dataset

    @staticmethod
    def createmodel(num_classes,
                    length_imsize, width_imsize, 
                    n_colors,
                    weightsfilepath=None, modelfilepath=None):
        # define model parameters
        model_params = {
            'num_classes': num_classes,
            'length_imsize': length_imsize,
            'width_imsize': width_imsize,
            'n_colors': n_colors,
        }
        model = EZNet(**model_params)

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
                    outputdir, expname, device=None):
        if device is None:
            devices = super(MarccHPC, MarccHPC).get_available_gpus()
        if len(devices) > 1:
            print("Let's use {} GPUs!".format(len(devices)))
            # make the model parallel
            model = multi_gpu_model(model, gpus=len(devices))

        trainer = EZNetTrainer(model=model, 
                            num_epochs=num_epochs, 
                            batch_size=batch_size,
                            outputdir=outputdir)
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
        trainer.test(modelname=modelname)
        return trainer

