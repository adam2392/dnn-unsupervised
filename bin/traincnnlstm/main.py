import sys
sys.path.append('../../dnn/')
sys.path.append('../dnn/')
import os
import numpy as np

# Custom Built libraries
from model.nets.ieegcnn import iEEGCNN
from model.train import trainseq

# metrics for postprocessing of the results
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, \
    recall_score, classification_report, \
    f1_score, roc_auc_score
    
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

if __name__ == '__main__':
    outputdatadir = str(sys.argv[1])
    tempdatadir = str(sys.argv[2])
    traindatadir = str(sys.argv[3])
    if not os.path.exists(outputdatadir):
        os.makedirs(outputdatadir)
    if not os.path.exists(tempdatadir):
        os.makedirs(tempdatadir)

    # load in model and weights -> NEED TO ADAPT OUTPUTDATADIR to get the correct weights! (version exp)
    weightsfile = os.path.join(outputdatadir, 'final_weights.h5')
    modelfile = os.path.join(outputdatadir, 'cnn_model.json')
    modelname = '2dcnn-lstm'
    pattraindir = os.path.join(traindatadir, 'realtng')
    
    # list of patients to train on
    listofpats_train = [
                    'id001',
                    'id002', 
                    'id008', 
                    # 'id010', 
                    'id011', 
                    'id012', 
                    'id013'
                ]
    listofpats_test = [
                    'id010'
                ]
    # alldatafile = os.path.join(traindatadir, 'realtng', 'allimages.npz')
    ##################### PARAMETERS FOR NN ####################
    imsize=32
    n_colors =3 
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
    ##################### TRAINING FOR NN ####################
    numtimesteps = 10
    batch_size = 32
    NUM_EPOCHS = 100
    AUGMENT = True

    cnn_trainer = trainseq.TrainSeq(dnnmodel, batch_size, numtimesteps, NUM_EPOCHS, AUGMENT)

    # configure, load generator and load training/testing data
    cnn_trainer.configure(tempdatadir)
    cnn_trainer.loadgenerator()
    cnn_trainer.loaddirofdata(pattraindir, listofpats_train, LOAD=True)
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

