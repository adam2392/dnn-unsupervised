import sys
sys.path.append('./dnn/')

import model.ieeg_cnn_rnn
import model.train

import processing.util as util

import keras
from keras.callbacks import ModelCheckpoint
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

def poly_decay(epoch, NUM_EPOCHS, INIT_LR):
    # initialize the maximum number of epochs, base learning rate,
    # and power of the polynomial
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0
    # compute the new learning rate based on polynomial decay
    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
    # return the new learning rate
    return alpha

def loadmodel(ieegdnn, **kwargs):
    if model=='cnn':
        # VGG-12 style later
        vggcnn = ieegdnn._build_2dcnn(w_init=w_init, n_layers=n_layers, 
                                      poolsize=poolsize, filter_size=filtersize)
        vggcnn = ieegdnn._build_seq_output(vggcnn, size_fc, DROPOUT)

from sklearn.metrics import roc_auc_score
class Histories(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        y_pred = self.model.predict(self.model.validation_data[0])
        self.aucs.append(roc_auc_score(self.model.validation_data[1], y_pred))
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return

def normalize(images):
    '''
    Use a function to normalize across the frequency bands of the images tensor
    [x, 32, 32, 4]
    '''
    pass

if __name__ == '__main__':
    traindatadir = str(sys.argv[1])
    tempdatadir = str(sys.argv[2])
    
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=cuda_dev

    ##################### PARAMETERS FOR NN ####################
    # image parameters #
    imsize=32
    numfreqs = 5
    numclasses = 2 

    # layer parameters #
    w_init = None       # weight intializers for all layers
    n_layers = (4,2,1)  # num of convolutional layers in sequence
    poolsize = (2,2)    # maxpooling size
    filtersize = (3,3)  # filter size
    size_fc = 512       # number of memory units to use in LSTM

    # fully connected output #
    size_fc = 512       # size of fully connected layers
    DROPOUT = False     # should we use Hinton Dropout method?

    # define number of epochs and batch size
    NUM_EPOCHS = 100
    batch_size = 32 # or 64... or 24

    ieegdnn = model.ieeg_cnn_rnn.IEEGdnn(imsize=imsize, 
                                        n_colors=numfreqs,
                                        num_classes=numclasses)

    ##################### INPUT DATA FOR NN ####################
    image_filepath = os.path.join(traindatadir, 'trainimages.npy')
    ylabel_filepath = os.path.join(traindatadir, 'trainlabels.npy')

    # define data filepath to images
    images = np.load(image_filepath)
    images = normalizeimages(images) # normalize the images for each frequency band
    # load the ylabeled data
    ylabels = np.load(ylabel_filepath)
    invert_y = 1 - ylabels
    ylabels = np.concatenate((ylabels, invert_y),axis=1)

    # format the data correctly 
    # (X_train, y_train), (X_val, y_val), (X_test, y_test) = datahandler.reformatinput(images, labels)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.33, random_state=42)
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")

    assert ylabels.shape[1]==2
    # assert images.shape
    ##################### TRAINING FOR NN ####################
    # VGG-12 style later
    vggcnn = ieegdnn._build_2dcnn(w_init=w_init, n_layers=n_layers, 
                                  poolsize=poolsize, filter_size=filtersize)
    vggcnn = ieegdnn._build_seq_output(vggcnn, size_fc, DROPOUT)
    sys.stdout.write("Created VGG12 Style CNN")
    print(vggcnn.summary())
    
    # load in previous weights
    # bestweightsfile = 'final_weights.hdf5'
    # vggcnn.load_weights(bestweightsfile)
    # sys.stdout.write("Created model and loaded weights from file")

    # initialize loss function, SGD optimizer and metrics
    loss = 'binary_crossentropy'
    optimizer = keras.optimizers.Adam(lr=0.001, 
                                    beta_1=0.9, 
                                    beta_2=0.999,
                                    epsilon=1e-08,
                                    decay=0.0)
    metrics = ['loss', 'accuracy', 'val_loss']

    # compile model
    cnn_config = ieegdnn.compile_model(vggcnn, 
                                    loss=loss, 
                                    optimizer=optimizer, 
                                    metrics=metrics)

    # construct the image generator for data augmentation and construct the set of callbacks
    aug = keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1,
                                                height_shift_range=0.1, 
                                                horizontal_flip=False,
                                                fill_mode="nearest")
    
    # checkpoint
    filepath=os.path.join(tempdatadir,"weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
                                        verbose=1, 
                                        save_best_only=True, 
                                        mode='max')
    callbacks = [checkpoint, poly_decay]
    INIT_LR = 5e-3
    G=1
    HH = vggcnn.fit_generator(
        aug.flow(X_train, y_train, batch_size=batch_size * G), # adds augmentation to data using generator
        validation_data=(X_test, y_test),  
        steps_per_epoch=len(X_train) // (batch_size * G),    #
        epochs=NUM_EPOCHS,
        callbacks=callbacks, verbose=2)

    # save final history object
    vggcnn.save(os.path.join(tempdatadir, 'final_weights.h5'))
