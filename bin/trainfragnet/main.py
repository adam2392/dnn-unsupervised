'''
Script file for HPC for running a 1D fragnet.

'''

import sys
sys.path.append('../../dnn/')
sys.path.append('../dnn/')
import os
import numpy as np

# Custom Built libraries
from model.nets.fragilityaux import CNNFragility
from model.nets.ieegcnn import iEEGCNN
from model.train import traincnn
from model.train.fragaux.processdata import LabelData

import keras
# metrics for postprocessing of the results
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, \
    recall_score, classification_report, \
    f1_score, roc_auc_score


def mainmodel(traindatadir, rawdatadir):
    ##################### PARAMETERS FOR NN - CREATE NN ####################
    n_colors = 1
    num_classes = 2       # dimension of output predictions
    DROPOUT = True
    modeldim = 1   # (optional): dim of model (1,2,3)
    imsize = 1     # since this is a 1D cnn?
    numwins = 500  # use 500 windows for a fragility

    # list of patients to train on
    listofpats = [
        'jh105',
        'pt1', 'pt2', 'pt3',
        'pt8', 'pt11', 'pt13', 'pt15'
        'pt16', 'pt17',
        'la01',
        'la02', 'la03',
        'la05', 'la07',
        'ummc002', 'ummc003',
        'ummc004', 'ummc005',
        'ummc006',
    ]

    # 01: initialize a munger class to help format the data
    datamunger = LabelData(numwins, rawdatadir)  # feed in PCA size
    # load all the data based on patients
    datamunger.loaddirofdata(traindatadir, listofpats)
    datamunger.formatdata()   #
    # setup training scheme for data
    datamunger.trainingscheme(scheme='rand')

    # 02: initialize the convolutional auxiliary network
    # build the baseline CNN model
    cnn = iEEGCNN(imsize=numwins,
                  n_colors=n_colors,
                  num_classes=num_classes,
                  modeldim=modeldim,
                  DROPOUT=DROPOUT)
    cnn.buildmodel()
    # instantiate this current model
    cnn.summaryinfo()
    dnnmodel = cnn.model

    # print some debugging outputs to allow user to see how model is being trained
    print("Input of fragnet model is: ", cnn.model.input_shape)
    sys.stdout.write("We have %i datasets" % len(datamunger.datafilepaths))
    sys.stdout.write(datamunger.datafilepaths[0])
    print(numwins)
    print(len(datamunger.ylabels))
    print(len(datamunger.main_data))
    return dnnmodel, datamunger


def maintrain(dnnmodel, datamunger, outputdatadir, tempdatadir):
    modelname = '1dcnn'
    ##################### PARAMETERS FOR TRAINING - CREATE TRAINER ####################
    batch_size = 32
    NUM_EPOCHS = 300
    AUGMENT = True

    cnn_trainer = traincnn.TrainCNN(dnnmodel, batch_size, NUM_EPOCHS, AUGMENT)

    # load the data into the trainer and then begin training
    class_weight = datamunger.class_weight
    Xmain_train = datamunger.Xmain_train
    y_train = datamunger.y_train
    Xmain_test = datamunger.Xmain_test
    y_test = datamunger.y_test

    # configure, load generator and load training/testing data
    cnn_trainer.configure(tempdatadir)
    # cnn_trainer.loaddirs(traindatadir, testdatadir,
    #                      listofpats_train, listofpats_test)
    cnn_trainer.loadtrainingdata_vars(Xmain_train, y_train)
    cnn_trainer.loadtestingdata_vars(Xmain_test, y_test)
    cnn_trainer.train()

    print(len(cnn_trainer.Xmain_train))
    print(len(cnn_trainer.y_train))

    # print out summary info for the model and the training
    cnn_trainer.summaryinfo()

    # save model, final weights and the history object
    cnn_trainer.saveoutput(modelname=modelname, outputdatadir=outputdatadir)

    # get the history object as a result of training
    HH = cnn_trainer.HH
    return cnn_trainer

def maintest(dnnmodel, cnn_trainer):
    # get the testing data to run a final summary output
    X_test = cnn_trainer.X_test
    y_test = cnn_trainer.y_test

    ##################### INPUT DATA FOR NN ####################
    prob_predicted = dnnmodel.predict(X_test)
    ytrue = np.argmax(y_test, axis=1)
    y_pred = dnnmodel.predict_classes(X_test)

    sys.stdout.write(prob_predicted.shape)
    sys.stdout.write(ytrue.shape)
    sys.stdout.write(y_pred.shape)
    print("ROC_AUC_SCORES: ", roc_auc_score(y_test, prob_predicted))
    print('Mean accuracy score: ', accuracy_score(ytrue, y_pred))
    print('F1 score:', f1_score(ytrue, y_pred))
    print('Recall:', recall_score(ytrue, y_pred))
    print('Precision:', precision_score(ytrue, y_pred))
    print('\n clasification report:\n', classification_report(ytrue, y_pred))
    print('\n confusion matrix:\n', confusion_matrix(ytrue, y_pred))


if __name__ == '__main__':
    # read in the inputs - data directories
    # the output data directory - where to store the final model, weights
    outputdatadir = str(sys.argv[1])
    # the temporary data directory - storing the computations at intervals
    tempdatadir = str(sys.argv[2])
    # the training data directory - storing fragility maps
    traindatadir = str(sys.argv[3])
    # the testing data direcotry - stores the fragility maps we want to use
    # testdatadir = str(sys.argv[4])

    # the datadir where the original raw data is
    rawdatadir = str(sys.argv[4])
    patient = str(sys.argv[5])

    # create the output and temporary saving directories
    if not os.path.exists(outputdatadir):
        os.makedirs(outputdatadir)
    if not os.path.exists(tempdatadir):
        os.makedirs(tempdatadir)

    # create the dnn model and the data munger if necessary
    dnnmodel, datamunger = mainmodel(traindatadir, rawdatadir)
    trainer = maintrain(dnnmodel, datamunger, outputdatadir, tempdatadir)
    maintest(dnnmodel, trainer)
