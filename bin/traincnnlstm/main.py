import sys
sys.path.append('../../dnn/')
sys.path.append('../dnn/')
import os
import numpy as np

# Custom Built libraries
import model.ieeg_cnn_rnn
import model.train
import processing.util as util

from processing.generators.generatorfromfile import 

# deep learning, scientific computing libs
import keras
from keras.models import Sequential, Model
from keras.layers import LSTM, TimeDistributed
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import Callback

# preprocessing data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

# metrics for postprocessing of the results
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, \
    recall_score, classification_report, \
    f1_score, roc_auc_score

# utilitiy libs
import ntpath
import json
import pickle

from sklearn.utils import class_weight
    
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

    modelname = '2dcnn-lstm'
    modeljsonfile = os.path.join(outputdatadir, modelname+"_model.json")
    historyfile = os.path.join(outputdatadir, 'history_2dcnn.pkl')
    finalweightsfile = os.path.join(outputdatadir, 'final_weights' + '.h5')
    tempfilepath = os.path.join(tempdatadir,"weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5")

    alldatafile = os.path.join(traindatadir, 'realtng', 'allimages.npz')

    ##################### LOAD PRETRAINED CNN ####################
    # load in model and weights -> NEED TO ADAPT OUTPUTDATADIR to get the correct weights! (version exp)
    weightsfile = os.path.join(outputdatadir, 'final_weights.h5')
    modelfile = os.path.join(outputdatadir, 'cnn_model.json')
    # load json and create model
    json_file = open(modelfile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    # load in the fixed_cnn_model
    fixed_cnn_model = keras.models.model_from_json(loaded_model_json)
    fixed_cnn_model.load_weights(weightsfile)
    # remove the last 2 dense FC layers and freeze it
    fixed_cnn_model.pop()
    fixed_cnn_model.pop()
    fixed_cnn_model.trainable = False

    ##################### PARAMETERS FOR NN ####################
    # fully connected output #
    imsize=32
    n_colors = 4
    num_timewins = 30
    size_mem = 120
    size_fc = 512       # size of fully connected layers
    DROPOUT = True     # should we use Hinton Dropout method?

    # define number of epochs and batch size
    NUM_EPOCHS = 100 # per dataset
    batch_size = 32 # or 64... or 24
    data_augmentation = True

    ##################### TRAINING FOR NN ####################
    ####### CNN-LSTM style ANN #######
    # create sequential model to get this all before the LSTM
    currmodel = Sequential()
    currmodel.add(TimeDistributed(fixed_cnn_model, 
                        input_shape=(num_timewins, imsize, imsize, n_colors)))
    currmodel.add(LSTM(units=size_mem, 
                        activation='relu', 
                        return_sequences=False))
    currmodel = ieegdnn._build_seq_output(model, size_fc, DROPOUT=True)
    currmodel = Model(inputs=model.input, outputs = model.output)

    print(currmodel.summary())
    if not os.path.exists(modeljsonfile):
        # serialize model to JSON
        model_json = currmodel.to_json()
        with open(modeljsonfile, "w") as json_file:
            json_file.write(model_json)
        print("Saved model to disk")

    # initialize loss function, SGD optimizer and metrics
    loss = 'binary_crossentropy'
    optimizer = RMSprop(lr=1e-4, 
                        rho=0.9, 
                        epsilon=1e-08,
                        decay=0.0)
    metrics = ['accuracy']

    modelconfig = currmodel.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    print(modelconfig)
    print("model input shape is: ", currmodel.input_shape)

    # This will do preprocessing and realtime data augmentation:
    datagen = keras.preprocessing.image.ImageDataGenerator(
                    featurewise_center=True,  # set input mean to 0 over the dataset
                    samplewise_center=False,  # set each sample mean to 0
                    featurewise_std_normalization=True,  # divide inputs by std of the dataset
                    samplewise_std_normalization=False,  # divide each input by its std
                    zca_whitening=False,      # apply ZCA whitening
                    rotation_range=3,         # randomly rotate images in the range (degrees, 0 to 180)
                    width_shift_range=0.02,    # randomly shift images horizontally (fraction of total width)
                    height_shift_range=0.02,   # randomly shift images vertically (fraction of total height)
                    horizontal_flip=False,    # randomly flip images
                    vertical_flip=False,      # randomly flip images
                    fill_mode='nearest')  

    # checkpoint
    checkpoint = ModelCheckpoint(tempfilepath, 
                                    monitor='val_acc', 
                                    verbose=1, 
                                    save_best_only=True, 
                                    mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8,
                          patience=10, min_lr=1e-8)
    testcheck = TestCallback()
    callbacks = [checkpoint] #, poly_decay]

    # ALL DEBUG BEFORE STARTING TRAINING
    print("Fixed CNN Model is: \n", fixed_cnn_model.summary())
    print("Model summary is: \n", currmodel.summary())
    print(modelconfig)
    print("model input shape is: ", currmodel.input_shape)

    ##################### INPUT DATA FOR NN ####################
    imagedir = os.path.join(traindatadir, 'image_2d')
    # get all the separate files to use for training:
    datafiles = []
    for root, dirs, files in os.walk(imagedir):
        for file in files:
            datafiles.append(os.path.join(root, file))
    sys.stdout.write('\nTraining on ' + str(len(datafiles)) + ' datasets!\n')

    # get a training/testing split on the data files generated
    trainfiles, testfiles = train_test_split(datafiles, test_size=0.33, random_state=42)

    # Parameters for our custom built generator
    params = {'dim_x': imsize,
              'dim_y': imsize,
              'dim_z': numfreqs,
              'batch_size': batch_size,
              'shuffle': True}
    datagen = processing.util.DataGenerator(**params)
    training_generator = datagen.generate_fromdir(trainfiles)
    validation_generator = datagen.generate_fromdir(testfiles)
    
    HH = currmodel.fit_generator(generator=training_generator,
            steps_per_epoch=2000,
            epochs=NUM_EPOCHS,
            validation_data=validation_generator,
            shuffle=False,
            callbacks=callbacks, verbose=2)
    # save final history object
    currmodel.save(os.path.join(outputdatadir, 
                    'final_weights' + '.h5'))