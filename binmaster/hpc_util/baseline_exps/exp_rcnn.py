import sys
import os
sys.path.append('/scratch/users/ali39@jhu.edu/dnn-unsupervised/')
sys.path.append(os.path.expanduser('~/Documents/dnn-unsupervised/'))
import argparse
import shutil

from dnn.execute.hpc_rcnnmodel import MarccHPC

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
    testpat = 'id001_bt'
    output_data_dir = os.path.expanduser('~/Downloads')
    train_data_dir = os.path.expanduser('~/Downloads/tngpipeline/freq/fft_img/')
    test_data_dir = train_data_dir
    expname = 'test'

    cnn_model_dir = os.path.expanduser('~/Downloads/exp001/id001_bt/output/')
    weightsfile = os.path.join(cnn_model_dir, 'loobasecnn_final_weights.h5')
    modelfile = os.path.join(cnn_model_dir, 'loobasecnn_model.json')

def hpc_run(args):
    # read in the parsed arguments
    testpat = args.patient_to_loo
    log_data_dir = args.log_data_dir
    output_data_dir = args.output_data_dir
    train_data_dir = args.train_data_dir
    cnn_model_dir = args.model_dir
    test_data_dir = args.test_data_dir
    expname = args.expname

    weightsfile = os.path.join(cnn_model_dir, testpat, 'output', 'loobasecnn_final_weights.h5')
    modelfile = os.path.join(cnn_model_dir, testpat, 'output', 'loobasecnn_model.json')

    print("args are: ", args)
    print("Number of different patients: {}".format(len(training_patients)))

    # parameters for model
    modelname = 'loobasecnn'
    num_classes = 2
    data_procedure='loo'
    seqlen = 20
    # training parameters 
    num_epochs = 150
    batch_size = 32
    learning_rate = 1e-3 # np.linspace(1e-5, 1e-3, 10)

    # for testpat in all÷_patients:
    testpatdir = os.path.join(output_data_dir, testpat)
    print("Our maint directory to save for loo exp: ", testpatdir)
    print(train_data_dir, test_data_dir)

    # initialize hpc trainer object
    hpcrun = MarccHPC()
    # get the datasets
    train_dataset, test_dataset = hpcrun.load_data(train_data_dir, test_data_dir, 
                                        seqlen=seqlen,
                                        data_procedure=data_procedure, 
                                        testpat=testpat, 
                                        training_pats=training_patients)
    # get the image size and n_colors from the datasets
    imsize = train_dataset.imsize
    n_colors = train_dataset.n_colors

    # create model
    model = hpcrun.createmodel(num_classes=num_classes, seqlen=seqlen//2, 
                            imsize=imsize, n_colors=n_colors,
                            weightsfile=weightsfile, modelfile=modelfile)
    # extract the actual model from the object
    model = model.net    
    # train model
    trainer = hpcrun.trainmodel(model=model, num_epochs=num_epochs, batch_size=batch_size, 
                        train_dataset=train_dataset, test_dataset=test_dataset,
                        testpatdir=testpatdir,  expname=expname)
    # test and save model
    trainer = hpcrun.testmodel(trainer, modelname)

    print("Image size is {} with {} colors".format(imsize, n_colors))
    print("Model is: {}".format(model))
    print("Model summary: {}".format(model.summary()))
    
if __name__ == '__main__':
    args = parser.parse_args()
    
    hpc_run(args)