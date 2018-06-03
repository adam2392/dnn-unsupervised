import numpy as np
import torch
import torch.nn as nn

# Custom Built libraries
import dnn_pytorch
from dnn_pytorch.io.read_dataset import Reader
from dnn_pytorch.base.dataset.fftdataset import FFT2DImageDataset
import dnn_pytorch.base.constants.model_constants as constants
from dnn_pytorch.models.nets.cnn import ConvNet
from dnn_pytorch.models.trainers.trainer import CNNTrainer

# preprocessing data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

# metrics for postprocessing of the results
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, \
    recall_score, classification_report, \
    f1_score, roc_auc_score
    
def createmodel(num_classes, imsize, n_colors):
    # 1) create the model - cnn
    model = ConvNet(num_classes=num_classes,
                    imsize=imsize,
                    n_colors=n_colors)
    model.buildcnn()
    return model

def trainmodel(model, train_dataset, test_dataset, logdatadir, outputdatadir, device=None):
    # training parameters 
    num_epochs = 100
    batch_size = 64

    if device is None:
        # Device configuration
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
          print("Let's use", torch.cuda.device_count(), "GPUs!")
          model = nn.DataParallel(model)

    # MOVE THE MODEL ONTO THE DEVICE WE WANT
    model = model.to(device)

    print(device)

    # 1) create the trainer
    trainer = CNNTrainer(model, num_epochs, batch_size, device=device, explogdir=logdatadir)
    trainer.composedatasets(train_dataset, test_dataset)
    trainer.run_config()

    print(trainer)
    # trainer.train()
    return trainer

def testmodel(trainer, resultfilename):
    trainer.save(resultfilename=resultfilename)
    return trainer

def load_data(traindir, testdir, data_procedure='loo', testpat=None):
    '''
    If LOO training, then we have to trim these into 
    their separate filelists
    '''

    # initialize reader to get the training/testing data
    reader = Reader()
    reader.loadbydir(traindir, testdir, procedure=data_procedure, testname=testpat)
    reader.loadfiles(mode=constants.TRAIN)
    reader.loadfiles(mode=constants.TEST)

    # create the dataset objects
    train_X = reader.X_train
    train_y = reader.y_train
    test_X = reader.X_test
    test_y = reader.y_test

    # create the loaders
    train_dataset = FFT2DImageDataset(train_X, train_y, mode=constants.TRAIN, transform=True)
    test_dataset = FFT2DImageDataset(test_X, test_y,  mode=constants.TEST, transform=None)
    return train_dataset, test_dataset

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