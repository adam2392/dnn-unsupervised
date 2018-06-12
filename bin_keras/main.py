from __future__ import print_function
print("inside main")
import sys
import os
sys.path.append('/scratch/users/ali39@jhu.edu/dnn-unsupervised/')
sys.path.append('../../')
import argparse
from run_util import *
import shutil

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
# all_patients = [
#     'id001_bt',
#     'id002_sd',
#     'id003_mg', 'id004_bj', 'id005_ft',
#     'id006_mr', 'id007_rd', 'id008_dmc',
#     'id009_ba', 'id010_cmn', 'id011_gr',
#     'id013_lk', 'id014_vc', 'id015_gjl',
#     'id016_lm', 'id017_mk', 'id018_lo', 'id020_lma']

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
    modelname = 'loobasecnn'
    num_classes = 2
    data_procedure='loo'
    # training parameters 
    num_epochs = 150
    batch_size = 32

    # for testpat in all÷_patients:
    testpatdir = os.path.join(output_data_dir, testpat)
    print("Our maint directory to save for loo exp: ", testpatdir)
    print(train_data_dir, test_data_dir)
    # get the datasets
    train_dataset, test_dataset = load_data(train_data_dir, test_data_dir, 
                            data_procedure=data_procedure, 
                            testpat=testpat, training_pats=training_patients)

    # get the image size and n_colors from the datasets
    imsize = train_dataset.imsize
    n_colors = train_dataset.n_colors
            
    # create model
    model = createmodel(num_classes, imsize, n_colors)
    print("Image size is {} with {} colors".format(imsize, n_colors))
    print("Model is: {}".format(model))
    
    # train model
    trainer = trainmodel(model, train_dataset, test_dataset,
                        testpatdir=testpatdir,  expname=expname)
    # test and save model
    trainer = testmodel(trainer, modelname)

if __name__ == '__main__':
    args = parser.parse_args()
    
    hpc_run(args)