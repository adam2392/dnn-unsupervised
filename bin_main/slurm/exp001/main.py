import sys
sys.path.append('../')
sys.path.append('../../../')
import os
import numpy as np

# Custom Built libraries
import dnn_pytorch
from dnn_pytorch.models.trainer import Trainer
from dnn_pytorch.models.cnn import ConvNet, Train

from dnn_pytorch.io.read_dataset import Reader
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

def trainmodel(model, train_dataset, test_dataset):
    # training parameters 
    num_epochs = 100
    batch_size = 64

    # 1) create the trainer
    trainer = Trainer(model, num_epochs, batch_size)
    trainer.composedatasets(train_dataset, test_dataset)
    trainer.train()
    return trainer

def testmodel(trainer):
    trainer.test()
    return trainer

def load_data(traindir, testdir):
    # trainfiles = reader.trainfilepaths
    # testfiles = 

    # initialize reader to get the training/testing data
    reader = Reader()
    reader.loadbydir(traindir, testdir)
    reader.loadfiles(mode=mode=constants.TRAIN,)
    reader.loadfiles(mode=mode=constants.TEST,)

    # create the dataset objects
    train_X = reader.X_train
    train_y = reader.y_train
    test_X = reader.X_test
    test_y = reader.y_test
    train_dataset = FFT2DImageDataset(train_X, train_y, mode=constants.TRAIN, transform=True)
    test_dataset = FFT2DImageDataset(test_X, test_y,  mode=constants.TEST, ransform=None)
    return train_dataset, test_dataset

if __name__ == '__main__':
    # outputdatadir = str(sys.argv[1])    # output for data dir
    # logdatadir = str(sys.argv[2])      # the temp data dire
    # datadir = str(sys.argv[3])          # the training data directory
    # testdatadir = str(sys.argv[4])      # the test data directory
    # patient = str(sys.argv[5])

    '''
    For testing locally
    '''
    outputdatadir = './output/'
    logdatadir = './logs/'
    testdatadir = './'
    datadir = './'
    patient = ''
    traindatadir = os.path.join(datadir, './')

    num_classes = 2

    # create the output and temporary saving directories
    # if not os.path.exists(outputdatadir):
    #     os.makedirs(outputdatadir)
    # if not os.path.exists(logdatadir):
    #     os.makedirs(logdatadir)

    # get the datasets
    train_dataset, test_dataset = load_data(traindir, testdir)
    # get the image size and n_colors from the datasets
    imsize = train_dataset.imsize
    n_colors = train_dataset.n_colors
    print("Image size is {} with {} colors".format(imsize, n_colors))
    
    # create model
    model = createmodel(num_classes, imsize, n_colors)

    # train model
    trainer = trainmodel(model, train_dataset, test_dataset)

    # test model
    trainer = testmodel(model)
