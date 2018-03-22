import sys
sys.path.append('../../dnn/')
sys.path.append('../dnn/')
import os
import numpy as np

print(sys.path)

# Custom Built libraries
from model.nets.ieegcnn import iEEGCNN
from model.train import traincnn

# preprocessing data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

# metrics for postprocessing of the results
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, \
    recall_score, classification_report, \
    f1_score, roc_auc_score

def mainmodel():
    ##################### PARAMETERS FOR NN - CREATE NN ####################
    imsize=32               # the size of the square image
    n_colors =4             # the #channels in convnet
    num_classes=2           # output dimension
    DROPOUT=True            # use DROPOUT?

    modeldim=2              # (optional): dim of model (1,2,3)

    # build the baseline CNN model
    cnn = iEEGCNN(imsize=imsize,
                  n_colors=n_colors, 
                  num_classes=num_classes, 
                  modeldim=modeldim, 
                  DROPOUT=DROPOUT)
    cnn.buildmodel()
    # instantiate this current model
    dnnmodel = cnn.model

    print("Input shape for the model is: ", dnnmodel.input_shape)
    print("Created VGG12 Style CNN")
    print(dnnmodel.summary())
    return dnnmodel
    
def maintrain(dnnmodel, outputdatadir, tempdatadir, traindatadir, testdatadir, patient):
    modelname = '2dcnn'
    # list of patients to train on
    listofpats_train = [
                    'id001_ac',
                    'id002_cj', 
                    'id008_gc', 
                    'id010_js', 
                    'id011_ml', 
                    'id012_pc', 
                    'id013_pg'
                    ]
    listofpats_test = [
                    'id001_ac',
                    'id002_cj', 
                    'id008_gc', 
                    'id010_js', 
                    'id011_ml', 
                    'id012_pc', 
                    'id013_pg'
                    ]

    listofpats_train.remove(patient)
    listofpats_test = [patient]
    ##################### PARAMETERS FOR TRAINING - CREATE TRAINER ####################
    batch_size = 32
    NUM_EPOCHS = 300
    AUGMENT = True

    cnn_trainer = traincnn.TrainCNN(dnnmodel, batch_size, NUM_EPOCHS, AUGMENT)

    # configure, load generator and load training/testing data
    cnn_trainer.configure(tempdatadir)
    cnn_trainer.loaddirs(traindatadir, testdatadir, 
                        listofpats_train, listofpats_test)
    cnn_trainer.loadtrainingdata()
    cnn_trainer.loadtestdata()
    cnn_trainer.train()

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

    print(prob_predicted.shape)
    print(ytrue.shape)
    print(y_pred.shape)
    print("ROC_AUC_SCORES: ", roc_auc_score(y_test, prob_predicted))
    print('Mean accuracy score: ', accuracy_score(ytrue, y_pred))
    print('F1 score:', f1_score(ytrue, y_pred))
    print('Recall:', recall_score(ytrue, y_pred))
    print('Precision:', precision_score(ytrue, y_pred))
    print('\n clasification report:\n', classification_report(ytrue, y_pred))
    print('\n confusion matrix:\n',confusion_matrix(ytrue, y_pred))

if __name__ == '__main__':
    outputdatadir = str(sys.argv[1])            # output for data dir
    tempdatadir = str(sys.argv[2])              # the temp data dire
    traindatadir = str(sys.argv[3])             # the training data directory
    testdatadir = str(sys.argv[4])              # the test data directory
    patient = str(sys.argv[5])
    '''
    outputdatadir = ''
    tempdatadir = ''
    traindatadir = ''
    testdatadir = ''
    '''
    # create the output and temporary saving directories
    if not os.path.exists(outputdatadir):
        os.makedirs(outputdatadir)
    if not os.path.exists(tempdatadir):
        os.makedirs(tempdatadir)

    dnnmodel = mainmodel()
    trainer = maintrain(dnnmodel, outputdatadir, tempdatadir, traindatadir, testdatadir, patient)
    maintest(dnnmodel, trainer)