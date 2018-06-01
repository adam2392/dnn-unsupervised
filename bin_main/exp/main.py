import numpy as np
import torch
import torch.nn as nn

# Custom Built libraries
import dnn_pytorch
from dnn_pytorch.models.trainers.trainer import Trainer
from dnn_pytorch.models.nets.cnn import ConvNet

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
print("inside main")
import sys
import os
sys.path.append('/scratch/users/ali39@jhu.edu/dnn-unsupervised/')
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('train_data_dir',
                    help="Directory containing the dataset(s)")
parser.add_argument('test_data_dir',
                    help="Directory containing the dataset(s)")
parser.add_argument('output_data_dir', default='/scratch/users/ali39@jhu.edu/data/dnn/output/', 
                    help="Directory to save logs")
parser.add_argument('log_data_dir', default='/scratch/users/ali39@jhu.edu/data/dnn/logs/', 
                    help="Directory to save logs")
parser.add_argument('patient_to_loo', default='id001_bt',
                    help="Patient to leave one out on.")
parser.add_argument('expname', default='_exp_default', 
                    help="name of the experiment name")
parser.add_argument('--model_dir', default='experiments/base_model', 
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', 
                    help="name of the file in --model_dir \
                     containing weights to load")
    
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
    trainer = Trainer(model, num_epochs, batch_size, device=device, explogdir=logdatadir)
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

def local_test(args):
    train_data_dir="/scratch/users/ali39@jhu.edu/data/dnn/traindata_fft/realtng/"
    test_data_dir="/scratch/users/ali39@jhu.edu/data/dnn/traindata_fft/realtng/"
    patient='id001_bt'

def hpc_run(args):
    testpat = args.patient_to_loo
    log_data_dir = args.log_data_dir
    output_data_dir = args.output_data_dir
    train_data_dir = args.train_data_dir
    test_data_dir = args.test_data_dir
    expname = args.expname

    log_data_dir='./'
    output_data_dir='./'
    # train_data_dir=os.path.expanduser("~/Downloads/tngpipeline/freqimg/fft/")
    # test_data_dir=os.path.expanduser("~/Downloads/tngpipeline/freqimg/fft/")

    num_classes = 2
    data_procedure='loo'

    logdatadir = os.path.join(log_data_dir, expname, patient)
    outputdatadir = os.path.join(output_data_dir, expname, patient)
    # create the output and temporary saving directories
    if not os.path.exists(outputdatadir):
        os.makedirs(outputdatadir)
    if not os.path.exists(logdatadir):
        os.makedirs(logdatadir)

    # get the datasets
    train_dataset, test_dataset = load_data(train_data_dir, test_data_dir, data_procedure=data_procedure, testpat=testpat)
    # get the image size and n_colors from the datasets
    imsize = train_dataset.imsize
    n_colors = train_dataset.n_colors
    print("Image size is {} with {} colors".format(imsize, n_colors))
    print(train_data_dir)
    print(test_data_dir)
    
    # create model
    model = createmodel(num_classes, imsize, n_colors)
    print(model)
    
    # train model
    trainer = trainmodel(model, train_dataset, test_dataset, logdatadir, outputdatadir)

    # # test model
    resultfilename = '{}_endmodel.ckpt'.format(testpat)
    trainer = testmodel(trainer, resultfilename)

if __name__ == '__main__':
    print("Inside main!")
    args = parser.parse_args()
    print(args)
    
    hpc_run(args)
    # local_test(args)