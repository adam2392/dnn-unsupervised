import sys
import os
import numpy as np
sys.path.append('/scratch/users/ali39@jhu.edu/dnn-unsupervised/')
import argparse
import shutil
sys.path.append(os.path.expanduser('~/Documents/dnn-unsupervised/'))
from dnn.execute.hpc_eznetmodel import MarccHPC

from keras.datasets import mnist
from dnn.io.dataloaders.base import TrainDataset, TestDataset

parser = argparse.ArgumentParser()
parser.add_argument('train_data_dir', default='./',
                    help="Directory containing the dataset(s)")
parser.add_argument('test_data_dir', default='./',
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

# TODO: pass this list into the models to allow it to know
# how to select directories for training
training_patients = [
    # updated tngpipeline
    'id001_bt',
    'id002_sd',
    'id003_mg', 'id004_bj', 'id005_ft',
    'id006_mr', 'id007_rd', 'id008_dmc',
    'id009_ba', 'id010_cmn', 'id011_gr',
    'id013_lk', 'id014_vc', 'id015_gjl',
    'id016_lm', 'id017_mk', 'id018_lo', 'id020_lma' ,
    'id021_jc', 'id022_te', 'id023_br',

    # old tngpipeline
    'id001_ac', 'id002_cj', 'id008_gc', 'id010_js', 'id011_ml',
    'id013_pg', 
]

def local_run(args):
    # read in the parsed arguments
    testpat = 'id001_bt'
    output_data_dir = os.path.expanduser('~/Downloads')
    train_data_dir = os.path.expanduser('~/Downloads/outputfreqimg/win5000_step2500/stft/')
    test_data_dir = train_data_dir
    expname = 'test'

def format_mnist():
    # get the datasets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # hardcode into binary classification 
    y_train_binary = (y_train == 5).astype(np.int)
    y_test_binary = (y_test == 5).astype(np.int)
    
    train_dataset = TrainDataset()
    test_dataset = TestDataset()
    
    traindataset.X_train = x_train
    traindataset.y_train = y_train_binary 

    testdataset.X_test = x_test
    testdataset.y_test = y_test_binary
    return train_dataset, testdataset

def hpc_run(args):
    # read in the parsed arguments
    testpat = args.patient_to_loo
    log_data_dir = args.log_data_dir
    output_data_dir = args.output_data_dir
    train_data_dir = args.train_data_dir
    test_data_dir = args.test_data_dir
    expname = args.expname

    print("args are: ", args)
    print("Number of different patients: {}".format(len(training_patients)))

    # parameters for model
    modelname = 'eznet'
    num_classes = 2
    data_procedure='loo'
    # training parameters 
    num_epochs = 150
    batch_size = 32
    learning_rate = 5e-4  # np.linspace(1e-5, 1e-3, 10)

    # for testpat in all÷_patients:
    testpatdir = os.path.join(output_data_dir, testpat)
    print("Our maint directory to save for loo exp: ", testpatdir)
    print(train_data_dir, test_data_dir)

    # initialize hpc trainer object
    hpcrun = MarccHPC()

    ########################## 1. LOAD DATA ##########################
    train_dataset, test_dataset = format_mnist()
    # get the image size and n_colors from the datasets
    length_imsize = 28
    width_imsize = 28
    width_imsize = 28*28
    length_imsize = 1
    n_colors = 1
            
    print("inputsize is: ", width_imsize, length_imsize, n_colors)
    ########################## 2. CREATE MODEL  ##########################
    # create model
    model = hpcrun.createmodel(num_classes, length_imsize, width_imsize, n_colors)
    # extract the actual model from the object
    model = model.net    
    
    ########################## 3. TRAIN MODEL ##########################
    # train model
    trainer = hpcrun.trainmodel(model=model, num_epochs=num_epochs, 
                        batch_size=batch_size, 
                        train_dataset=train_dataset, test_dataset=test_dataset,
                        outputdir=testpatdir, expname=expname)
    
    ########################## 4. TEST MODEL  ##########################
    # test and save model
    trainer = hpcrun.testmodel(trainer, modelname)

    print("Image size is {} with {} colors".format(imsize, n_colors))
    print("Model is: {}".format(model))
    print("Model summary: {}".format(model.summary()))

if __name__ == '__main__':
    args = parser.parse_args()
    # local_run(args)
    hpc_run(args)
