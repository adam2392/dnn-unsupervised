import sys
sys.path.append('../../dnn/')
sys.path.append('../dnn/')
import os
import numpy as np

# Custom Built libraries
from model.nets.fragilityaux import CNNFragility
from model.train import traincnnaux
from model.train.fragaux.processdata import SplitData

import keras
# metrics for postprocessing of the results
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, \
    recall_score, classification_report, \
    f1_score, roc_auc_score

if __name__ == '__main__':
    # read in the inputs - data directories
    # the output data directory - where to store the final model, weights
    outputdatadir = str(sys.argv[1])
    # the temporary data directory - storing the computations at intervals
    tempdatadir = str(sys.argv[2])
    # the training data directory - storing fragility maps
    traindatadir = str(sys.argv[3])
    # the datadir where the original raw data is
    rawdatadir = str(sys.argv[4])
    # rawdatadir = '/Volumes/ADAM LI/pydata/converted/'

    if not os.path.exists(outputdatadir):
        os.makedirs(outputdatadir)
    if not os.path.exists(tempdatadir):
        os.makedirs(tempdatadir)

    # load in model and weights -> NEED TO ADAPT OUTPUTDATADIR to get the correct weights! (version exp)
    weightsfile = os.path.join(outputdatadir, 'fragfinal_weights.h5')
    modelfile = os.path.join(outputdatadir, 'fragcnn_model.json')

    modelname = '2dfragnet'

    # imsize=30    # the size of the PCA that you perform on the rest of the fragility map
    n_colors = 1
    num_classes = 2
    DROPOUT = True

    pcsize = 40
    numwins = 500
    NUM_EPOCHS = 100
    AUGMENT = True
    batch_size = 16

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
        #     'ummc008',
        #     'id003', 'id004', 'id005',
        #     'id008', 'id010', 'id013'
        #     'ummc001', 'ummc007', 'ummc009'
    ]

    # 01: initialize a munger class to help format the data
    datamunger = SplitData(pcsize, numwins, rawdatadir)  # feed in PCA size
    # load all the data based on patients
    datamunger.loaddirofdata(traindatadir, listofpats)
    datamunger.formatdata()   #
    # setup training scheme for data
    datamunger.trainingscheme(scheme='rand')

    print("We have %i datasets" % len(datamunger.datafilepaths))
    print(datamunger.datafilepaths[0])
    print(numwins)
    print(len(datamunger.ylabels))
    print(len(datamunger.aux_data))
    print(len(datamunger.main_data))

    # 02: initialize the convolutional auxiliary network
    cnn = CNNFragility(numwins=numwins,
                       imsize=imsize,
                       n_colors=n_colors,
                       num_classes=num_classes,
                       num_timesteps=num_timesteps,
                       DROPOUT=DROPOUT)
    cnn.buildmodel()
    cnn.summaryinfo()
    print(cnn.model.input_shape)

    # 03: initialize the trainer
    trainer = traincnnaux.TrainFragAux(
        cnn.model, numwins, batch_size, NUM_EPOCHS, AUGMENT)

    # load the data into the trainer and then begin training
    class_weight = datamunger.class_weight
    Xmain_train = datamunger.Xmain_train
    Xaux_train = datamunger.Xaux_train
    y_train = datamunger.y_train
    trainer.loadformatteddata(Xmain_train, Xmain_test, Xaux_train, Xaux_test,
                              y_train, y_test, class_weight)
    trainer.configure(tempdatadir)
    trainer.loadgenerator()
    trainer.train()

    # print out summary info for the model and the training
    # trainer.summaryinfo()

    # save model, final weights and the history object
    trainer.saveoutput(modelname=modelname, outputdatadir=outputdatadir)

    # get the history object as a result of training
    HH = trainer.HH
    dnnmodel = trainer.dnnmodel

    # get the testing data to run a final summary output
    Xmain_test = trainer.Xmain_test
    Xaux_test = trainer.Xaux_test
    y_test = trainer.y_test

    ##################### INPUT DATA FOR NN ####################
    prob_predicted = dnnmodel.predict((Xmain_test, Xaux_test))
    ytrue = np.argmax(y_test, axis=1)
    y_pred = dnnmodel.predict_classes((Xmain_test, Xaux_test))
    print(prob_predicted.shape)
    print(ytrue.shape)
    print(y_pred.shape)
    print("ROC_AUC_SCORES: ", roc_auc_score(y_test, prob_predicted))
    print('Mean accuracy score: ', accuracy_score(ytrue, y_pred))
    print('F1 score:', f1_score(ytrue, y_pred))
    print('Recall:', recall_score(ytrue, y_pred))
    print('Precision:', precision_score(ytrue, y_pred))
    print('\n clasification report:\n', classification_report(ytrue, y_pred))
    print('\n confusion matrix:\n', confusion_matrix(ytrue, y_pred))
