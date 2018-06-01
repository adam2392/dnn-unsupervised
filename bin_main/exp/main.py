from __future__ import print_function
print("inside main")
import sys
import os
sys.path.append('/scratch/users/ali39@jhu.edu/dnn-unsupervised/')
import argparse
import numpy as np
import torch
import torch.nn as nn
from run import *

parser = argparse.ArgumentParser()
parser.add_argument('train_data_dir',
                    help="Directory containing the dataset(s)")
parser.add_argument('test_data_dir',
                    help="Directory containing the dataset(s)")
parser.add_argument('--output_data_dir', default='/scratch/users/ali39@jhu.edu/data/dnn/output/', 
                    help="Directory to save logs")
parser.add_argument('--log_data_dir', default='/scratch/users/ali39@jhu.edu/data/dnn/logs/', 
                    help="Directory to save logs")
parser.add_argument('--patient_to_loo', default='id001_bt',
                    help="Patient to leave one out on.")
parser.add_argument('--expname', default='_exp_default', 
                    help="name of the experiment name")
parser.add_argument('--model_dir', default='experiments/base_model', 
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', 
                    help="name of the file in --model_dir \
                     containing weights to load")

def local_test(args):
    train_data_dir="/scratch/users/ali39@jhu.edu/data/dnn/traindata_fft/realtng/"
    test_data_dir="/scratch/users/ali39@jhu.edu/data/dnn/traindata_fft/realtng/"
    patient='id001_bt'

def hpc_run(args):
    # read in the parsed arguments
    testpat = args.patient_to_loo
    log_data_dir = args.log_data_dir
    output_data_dir = args.output_data_dir
    train_data_dir = args.train_data_dir
    test_data_dir = args.test_data_dir
    expname = args.expname

    # assign log directories and output for saving model training
    log_data_dir='./'
    output_data_dir='./'

    # parameters for model
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
    # print(sys.args)
    
    hpc_run(args)
    # local_test(args)