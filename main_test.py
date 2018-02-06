import sys
sys.path.append('./dnn/')

# Custom Built libraries
import model.ieeg_cnn_rnn
import model.train
import processing.util as util

# deep learning, scientific computing libs
import keras
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, \
    recall_score, classification_report, \
    f1_score

# utilitiy libs
import ntpath
import json
import pickle

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def normalizeimage(imagetensor):
    # assumes imagetensor shape is [nsamps, width, height, nchans]
    temptensor = imagetensor.reshape((-1, 4))
    
    # get average/std and store temporary
    avg = np.mean(temptensor, axis=0, keepdims=True)
    std = np.std(temptensor, axis=0, keepdims=True)
    
    temptensor = np.subtract(temptensor, avg)
    temptensor = np.divide(temptensor,std)
    print(temptensor.shape)
    imagetensor = temptensor.reshape((-1, 32, 32, 4))
    return imagetensor


if __name__ == '__main__':
    outputdatadir = str(sys.argv[1])
    tempdatadir = str(sys.argv[2])
    traindatadir = str(sys.argv[3])

    if not os.path.exists(outputdatadir):
        os.makedirs(outputdatadir)
    if not os.path.exists(tempdatadir):
        os.makedirs(tempdatadir)

    ########### LOAD MODEL ##########
    # load in model and weights
    weightsfile = os.path.join(outputdatadir, 'tenpercentcorrupt/temp/_temp2dcnn/', 'weights-improvement-44-0.83.hdf5')
    modelfile = os.path.join(outputdatadir, 'tenpercentcorrupt/output/_final2dcnn/', '2dcnn_model.json')

    # load json and create model
    json_file = open(modelfile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    # load in the fixed_cnn_model
    currmodel = keras.models.model_from_json(loaded_model_json)
    currmodel.load_weights(weightsfile)

    # initialize loss function, SGD optimizer and metrics
    loss = 'binary_crossentropy'
    # loss = 'sparse_categorical_crossentropy'
    optimizer = keras.optimizers.Adam(lr=0.001, 
                                    beta_1=0.9, 
                                    beta_2=0.999,
                                    epsilon=1e-08,
                                    decay=0.0)
    metrics = ['accuracy']

    modelconfig = currmodel.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    print("model input shape is: ", currmodel.input_shape)

    ##################### INPUT DATA FOR NN ####################
    alldatafile = os.path.join(traindatadir, 'testdata', 'id001_ac_fft.npz')
    data = np.load(alldatafile)
    print(data.keys())
    images = data['image_tensor']
    metadata = data['metadata'].item()
    numfreqs = 4
    imsize = 32

    # reshape
    images = images.reshape((-1, numfreqs, imsize, imsize))
    images = images.swapaxes(1,3)

    # load the ylabeled data
    ylabels = metadata['ylabels']
    # invert_y = 1 - ylabels
    # ylabels = np.concatenate((ylabels, invert_y),axis=1)
    sys.stdout.write("\n\n Images and ylabels shapes are: \n\n")
    print(images.shape)
    print(ylabels.shape)
    sys.stdout.write("\n\n") 

    # assert the shape of the images
    assert images.shape[2] == images.shape[1]
    assert images.shape[2] == imsize
    assert images.shape[3] == numfreqs
    
    print(images.shape)
    images = images.astype("float32")
    
    print(images.shape)
    # augment data, or not and then trian the model!
    currmodel.predict(images)

    score = currmodel.evaluate(images, ylabels, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    predicted = currmodel.predict(images)
    ytrue = ylabels

    print(ytrue.shape)
    print(predicted.shape)

    # print('Mean accuracy score: ', accuracy_score(ytrue, predicted))
    # print('F1 score:', f1_score(ytrue, predicted))
    # print('Recall:', recall_score(ytrue, predicted))
    # print('Precision:', precision_score(ytrue, predicted))
    # print('\n clasification report:\n', classification_report(ytrue, predicted))
    # print('\n confusion matrix:\n',confusion_matrix(ytrue, predicted))

    print('\n\n Now normalizing \n\n')

    normimages = normalizeimage(images)
    print(normimages.shape)
    # augment data, or not and then trian the model!
    currmodel.predict(normimages)

    score = currmodel.evaluate(normimages, ylabels, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    predicted = currmodel.predict(images)
    ytrue = ylabels

    print(ytrue.shape)
    print(predicted.shape)
    print(predicted[0:5,:])

    # print('Mean accuracy score: ', accuracy_score(ytrue, predicted))
    # print('F1 score:', f1_score(ytrue, predicted))
    # print('Recall:', recall_score(ytrue, predicted))
    # print('Precision:', precision_score(ytrue, predicted))
    # print('\n clasification report:\n', classification_report(ytrue, predicted))
    # print('\n confusion matrix:\n',confusion_matrix(ytrue, predicted))



    #################### LOAD CORRECT MODEL #################
    # load in model and weights
    weightsfile = os.path.join(outputdatadir, '/temp/2dcnn/', 'weights-improvement-27-0.88.hdf5')
    modelfile = os.path.join(outputdatadir, '/output/final2dcnn/', '2dcnn_model.json')
    
    # load json and create model
    json_file = open(modelfile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    # load in the fixed_cnn_model
    currmodel = keras.models.model_from_json(loaded_model_json)
    currmodel.load_weights(weightsfile)

    # initialize loss function, SGD optimizer and metrics
    loss = 'binary_crossentropy'
    optimizer = keras.optimizers.Adam(lr=0.001, 
                                    beta_1=0.9, 
                                    beta_2=0.999,
                                    epsilon=1e-08,
                                    decay=0.0)
    metrics = ['accuracy']

    modelconfig = currmodel.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    print("model input shape is: ", currmodel.input_shape)

    # augment data, or not and then trian the model!
    currmodel.predict(images)

    score = currmodel.evaluate(images, ylabels, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    predicted = currmodel.predict(images)
    ytrue = ylabels
    # print('Mean accuracy score: ', accuracy_score(ytrue, predicted))
    # print('F1 score:', f1_score(ytrue, predicted))
    # print('Recall:', recall_score(ytrue, predicted))
    # print('Precision:', precision_score(ytrue, predicted))
    # print('\n clasification report:\n', classification_report(ytrue, predicted))
    # print('\n confusion matrix:\n',confusion_matrix(ytrue, predicted))

    print('\n\n Now normalizing \n\n')

    # augment data, or not and then trian the model!
    currmodel.predict(normimages)

    score = currmodel.evaluate(images, ylabels, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    predicted = currmodel.predict(images)
    ytrue = ylabels
    # print('Mean accuracy score: ', accuracy_score(ytrue, predicted))
    # print('F1 score:', f1_score(ytrue, predicted))
    # print('Recall:', recall_score(ytrue, predicted))
    # print('Precision:', precision_score(ytrue, predicted))
    # print('\n clasification report:\n', classification_report(ytrue, predicted))
    # print('\n confusion matrix:\n',confusion_matrix(ytrue, predicted))

    