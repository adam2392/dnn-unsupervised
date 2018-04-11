import sys
sys.path.append('../../dnn/')
sys.path.append('../dnn/')
import os
import numpy as np

# Custom Built libraries
from model.nets.ieegcnn import iEEGCNN
from model.nets.ieegseq import iEEGSeq
from model.train import trainseq

import keras

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

    # load in model and weights -> NEED TO ADAPT OUTPUTDATADIR to get the
    # correct weights! (version exp)
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
    ##################### PARAMETERS FOR NN ####################
    # CNN PARAMS
    imsize = 32
    n_colors = 3
    num_classes = 2
    modeldim = 2
    DROPOUT = True

    # LSTM PARAMS
    name = 'MIX'
    name = 'SAME'
    num_timewins = 5
    DROPOUT = True
    BIDIRECT = False
    cnnseq = iEEGSeq(name=name,
                     num_classes=num_classes,
                     num_timewins=num_timewins,
                     DROPOUT=DROPOUT,
                     BIDIRECT=BIDIRECT)
    LOAD = False
    if not LOAD:
        cnn = iEEGCNN(imsize=imsize,
                      n_colors=n_colors,
                      num_classes=num_classes,
                      modeldim=modeldim,
                      DROPOUT=DROPOUT)
        cnn.buildmodel()
        cnn.summaryinfo()

        print("EACH CNN MODEL INPUT IS: ", cnn.model.input_shape)
        fixed_cnn_model = cnn.model
    else:
        # LOAD A CNN NEURAL NETWORK FROM PREVIOUS TRAINS
        fixed_cnn_model = cnnseq.loadmodel(modelfile, weightsfile)

    sys.stdout.write("Created VGG12 Style CNN")
    # set weights to false
    fixed_cnn_model.trainable = False
    print(fixed_cnn_model.summary())
    print("Each CNN model input shape is: ", fixed_cnn_model.input_shape)

    # BUILD THE SEQUENTIAL MODEL - pass in the fixed_cnn model
    cnnseq.buildmodel(fixed_cnn_model)
    cnnseq.buildoutput()

    # print cnn-seq model's summary
    cnnseq.model.summary()

    ##################### TRAINING FOR NN ####################
    numtimesteps = 10
    batch_size = 32
    NUM_EPOCHS = 200
    AUGMENT = True

    seq_trainer = trainseq.TrainSeq(
        fixed_cnn_model, batch_size, numtimesteps, NUM_EPOCHS, AUGMENT)

    # configure, load generator
    seq_trainer.configure(tempdatadir)
    seq_trainer.loadgenerator()
    # load directory of data to compute file paths for the data we want per
    # patient
    seq_trainer.loaddirofdata(pattraindir, listofpats_train)
    # use the training data and loop through once to compute the class weights
    seq_trainer.compute_classweights()

    # run training
    seq_trainer.train()

    # print out summary info for the model and the training
    seq_trainer.summaryinfo()

    # save model, final weights and the history object
    seq_trainer.saveoutput(modelname=modelname, outputdatadir=outputdatadir)

    # get the history object as a result of training
    HH = seq_trainer.HH
    dnnmodel = cnn_trainer.dnnmodel
