import numpy as np
import os
import keras
from keras.utils.training_utils import multi_gpu_model

# Custom Built libraries
import dnn
from dnn.execute.hpc_basemodel import BaseHPC
import dnn.base.constants.model_constants as constants
from dnn.keras_models.nets.rcnn import RCNN 
from dnn.keras_models.trainers.rcnn import RCNNTrainer
from dnn.io.readerseqdataset import ReaderSeqDataset 

class MarccHPC(BaseHPC):
    '''
    An implementation specifcally for our MARCC HPC runner that
    impelements the basehpc functions.
    '''
    @staticmethod
    def load_data(trainfilepaths, testfilepaths, seqlen, 
            data_procedure='loo', testpat=None, training_pats=None):
        '''
        If LOO training, then we have to trim these into 
        their separate filelists
        '''
        # initialize reader to get the training/testing data
        reader = ReaderSeqDataset()
        # reader.load_filepaths(traindir, testdir, procedure=data_procedure, testname=testpat)
        reader.loadfiles_list(seqlen, filelist=trainfilepaths, mode=constants.TRAIN)
        reader.loadfiles_list(seqlen, filelist=testfilepaths, mode=constants.TEST)

        # create the dataset objects
        train_dataset = reader.train_dataset
        test_dataset = reader.test_dataset

        return train_dataset, test_dataset

    @staticmethod
    def createmodel(num_classes, seqlen, 
                    imsize, n_colors,
                    weightsfile, modelfile):
        # define model parameters
        model_params = {
            'name': 'SAME',
            'seqlen': seqlen,
            'num_classes': num_classes,
            'imsize': imsize,
            'n_colors': n_colors,
        }
        model = RCNN(**model_params)

        # load in old model
        model.loadmodel_file(modelfile, weightsfile)
        # build the overall model
        model.buildmodel(output=True)
        return model

    @staticmethod
    def trainmodel(model, num_epochs, batch_size, train_dataset, test_dataset, 
                        outputdir, expname, device=None):
        if device is None:
            devices = super(MarccHPC, MarccHPC).get_available_gpus()
        if len(devices) > 1:
            print("Let's use {} GPUs!".format(len(devices)))
            # make the model parallel
            model = multi_gpu_model(model, gpus=len(devices))

        trainer = RCNNTrainer(model=model, num_epochs=num_epochs, 
                            batch_size=batch_size,
                            outputdir=outputdir)
        trainer.composedatasets(train_dataset, test_dataset)
        trainer.configure()
        # Train the model
        trainer.train()
        print(model.net.summary())
        print("Training on {} ".format(device))
        print("Training object: {}".format(trainer))
        return trainer

    @staticmethod
    def testmodel(trainer, modelname):
        trainer.saveoutput(modelname=modelname)
        trainer.savemetricsoutput(modelname=modelname)
        return trainer

