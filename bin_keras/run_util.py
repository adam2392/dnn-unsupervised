import numpy as np
import os
import keras
from keras.utils.training_utils import multi_gpu_model

# Custom Built libraries
import dnn_keras
from dnn_pytorch.io.read_dataset import Reader
from dnn_pytorch.base.dataset.fftdataset import FFT2DImageDataset
import dnn_pytorch.base.constants.model_constants as constants
from dnn_pytorch.models.nets.cnn import ConvNet
from dnn_pytorch.models.trainers.trainer import CNNTrainer
  
from tensorflow.python.client import device_lib
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def load_data(traindir, testdir, data_procedure='loo', testpat=None):
    '''
    If LOO training, then we have to trim these into 
    their separate filelists
    '''
    pass

def createmodel(num_classes, imsize, n_colors):
    # 1) create the model - cnn
    model = ConvNet(num_classes=num_classes,
                    imsize=imsize,
                    n_colors=n_colors)
    model.buildcnn()
    model.buildoutput()
    return model

def trainmodel(model, num_epochs, batch_size, train_dataset, test_dataset, 
                    testpatdir, expname, device=None):
    if device is None:
        devices = get_available_gpus()
    if len(devices) > 0:
        print("Let's use {} GPUs!".format(len(devices)))
        # make the model parallel
        model = multi_gpu_model(model, gpus=len(devices))

    # MOVE THE MODEL ONTO THE DEVICE WE WANT
    model = model.to(device)
    
    # 1) create the trainer
    trainer = CNNTrainer(model, num_epochs, batch_size, 
                        device=device, testpatdir=testpatdir, expname=expname)
    trainer.composedatasets(train_dataset, test_dataset)
    trainer.run_config()

    print("Training on {} ".format(device))
    print("Training object: {}".format(trainer))
    trainer.train_and_evaluate()
    return trainer

def testmodel(trainer, resultfilename, historyfilename):
    pass
    return trainer

def localtest():
    '''
    For testing locally
    '''
    outputdatadir = './output/'
    logdatadir = './logs/'
    testdatadir = './'
    datadir = './'
    patient = ''
    traindatadir = os.path.join(datadir, './')