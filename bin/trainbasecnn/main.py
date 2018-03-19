import sys
sys.path.append('../../dnn/')
sys.path.append('../dnn/')
import os
import numpy as np

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

if __name__ == '__main__':
    outputdatadir = str(sys.argv[1])
    tempdatadir = str(sys.argv[2])
    traindatadir = str(sys.argv[3])
    if not os.path.exists(outputdatadir):
        os.makedirs(outputdatadir)
    if not os.path.exists(tempdatadir):
        os.makedirs(tempdatadir)

    # data dirs for training
    modelname = '2dcnn'
    # list of patients to train on
    listofpats_train = [
                    'id001',
                    'id002', 
                    # 'id008', 
                    'id010', 
                    'id011', 
                    'id012', 
                    'id013'
                ]
    listofpats_test = [
                    'id010'
                ]
    ##################### PARAMETERS FOR NN - CREATE NN ####################
    imsize=32
    n_colors =4
    num_classes=2
    modeldim=2
    DROPOUT=True

    cnn = iEEGCNN(imsize=imsize,
                  n_colors=n_colors, 
                  num_classes=num_classes, 
                  modeldim=modeldim, 
                  DROPOUT=DROPOUT)
    cnn.buildmodel()
    cnn.buildoutput()
    print(cnn.model.input_shape)
    sys.stdout.write("Created VGG12 Style CNN")

    # instantiate this current model
    dnnmodel = cnn.model

    print(dnnmodel.summary())
    print("model input shape is: ", dnnmodel.input_shape)

    ##################### PARAMETERS FOR TRAINING - CREATE TRAINER ####################
    batch_size = 32
    NUM_EPOCHS = 100
    AUGMENT = True
    cnn_trainer = traincnn.TrainCNN(dnnmodel, batch_size, NUM_EPOCHS, AUGMENT)

    # configure, load generator and load training/testing data
    cnn_trainer.configure(tempdatadir)
    cnn_trainer.loadgenerator()
    cnn_trainer.loaddirofdata(traindatadir, listofpats_train, LOAD=True)
    cnn_trainer.train()

    # print out summary info for the model and the training
    cnn.summaryinfo()
    cnn_trainer.summaryinfo()

    # save model, final weights and the history object
    cnn_trainer.saveoutput(modelname=modelname, outputdatadir=outputdatadir)

    # get the history object as a result of training
    HH = cnn_trainer.HH
    dnnmodel = cnn_trainer.dnnmodel

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
