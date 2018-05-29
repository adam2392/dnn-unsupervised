import sys
sys.path.append('../')
import os
import numpy as np
import torch
import torch.nn as nn

# Custom Built libraries
import dnn_pytorch
from dnn_pytorch.models.trainer import Trainer
from dnn_pytorch.models.cnn import ConvNet, Train

from dnn_pytorch.io.read_dataset import Reader
from dnn_pytorch.base.dataset.fftdataset import FFT2DImageDataset
import dnn_pytorch.base.constants.model_constants as constants

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
    return model

def trainmodel(model, train_dataset, test_dataset, logdatadir, outputdatadir):
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
    trainer = Trainer(model, num_epochs, batch_size, device=device, explogdir=expname)
    trainer.composedatasets(train_dataset, test_dataset)
    trainer.config()

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

    print(reader.trainfilepaths)
    print(reader.testfilepaths)

    # create the dataset objects
    train_X = reader.X_train
    train_y = reader.y_train
    test_X = reader.X_test
    test_y = reader.y_test

    # create the loaders
    train_dataset = FFT2DImageDataset(train_X, train_y, mode=constants.TRAIN, transform=True)
    test_dataset = FFT2DImageDataset(test_X, test_y,  mode=constants.TEST, ransform=None)
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

if __name__ == '__main__':
#     traindatadir="/scratch/users/ali39@jhu.edu/data/dnn/traindata_fft/expfull/"
# testdatadir="/scratch/users/ali39@jhu.edu/data/dnn/traindata_fft/realtng/"
# # logs for the training logs, and outputdata directory for final summary
# logdatadir="/scratch/users/ali39@jhu.edu/data/dnn/logs/$expname/"           
# outputdatadir="/scratch/users/al
    # ${outputdatadir} ${logdatadir} ${traindatadir} ${testdatadir} ${patient}
    outputdatadir = str(sys.argv[1])    # output for data dir
    logdatadir = str(sys.argv[2])      # the temp data dire
    traindatadir = str(sys.argv[3])          # the training data directory
    testdatadir = str(sys.argv[4])      # the test data directory
    patient = str(sys.argv[5])

    num_classes = 2
    data_procedure='loo'
    testpat = patient

    logdatadir = os.path.join(logdatadir, patient)
    outputdatadir = os.path.join(outputdatadir, patient)
    # create the output and temporary saving directories
    if not os.path.exists(outputdatadir):
        os.makedirs(outputdatadir)
    if not os.path.exists(logdatadir):
        os.makedirs(logdatadir)

    # get the datasets
    train_dataset, test_dataset = load_data(traindir, testdir, data_procedure=data_procedure, testpat=testpat)
    # get the image size and n_colors from the datasets
    imsize = train_dataset.imsize
    n_colors = train_dataset.n_colors
    print("Image size is {} with {} colors".format(imsize, n_colors))

    print(traindir)
    print(testdir)
    
    # create model
    model = createmodel(num_classes, imsize, n_colors)
    print(model)
    
    # train model
    trainer = trainmodel(model, train_dataset, test_dataset, logdatadir, outputdatadir)

    # # test model
    resultfilename = '{}_endmodel.ckpt'.format(patient)
    trainer = testmodel(trainer, resultfilename)

