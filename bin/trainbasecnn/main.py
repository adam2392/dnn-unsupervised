import sys
sys.path.append('../../dnn/')
sys.path.append('../dnn/')
import os
import numpy as np

# Custom Built libraries
import model.ieeg_cnn_rnn
import model.train
import processing.util as util

# deep learning, scientific computing libs
import keras
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

class TestCallback(Callback):
    def __init__(self):
        # self.test_data = test_data
        self.aucs = []

    def on_epoch_end(self, epoch, logs={}):
        # x, y = self.test_data
        x = self.model.validation_data[0]
        y = self.model.validation_data[1]

        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

        predicted = self.model.predict(x)
        self.aucs.append(roc_auc_score(y, predicted))

        predicted = self.model.predict_classes(x)
        ytrue = np.argmax(y, axis=1)
        print('Mean accuracy score: ', accuracy_score(ytrue, predicted))
        print('F1 score:', f1_score(ytrue, predicted))
        print('Recall:', recall_score(ytrue, predicted))
        print('Precision:', precision_score(ytrue, predicted))
        print('\n clasification report:\n', classification_report(ytrue, predicted))
        print('\n confusion matrix:\n',confusion_matrix(ytrue, predicted))
    
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def loadmodel(ieegdnn, **kwargs):
    if model=='cnn':
        # VGG-12 style later
        vggcnn = ieegdnn._build_2dcnn(w_init=w_init, n_layers=n_layers, 
                                      poolsize=poolsize, filter_size=filtersize)
        vggcnn = ieegdnn._build_seq_output(vggcnn, size_fc, DROPOUT)


if __name__ == '__main__':
    outputdatadir = str(sys.argv[1])
    tempdatadir = str(sys.argv[2])
    traindatadir = str(sys.argv[3])
    if not os.path.exists(outputdatadir):
        os.makedirs(outputdatadir)
    if not os.path.exists(tempdatadir):
        os.makedirs(tempdatadir)

    modelname = '2dcnn'
    modeljsonfile = os.path.join(outputdatadir, modelname+"_model.json")
    historyfile = os.path.join(outputdatadir, 'history_2dcnn.pkl')
    finalweightsfile = os.path.join(outputdatadir, 'final_weights' + '.h5')
    tempfilepath = os.path.join(tempdatadir,"weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5")

    alldatafile = os.path.join(traindatadir, 'realtng', 'allimages.npz')
    ##################### PARAMETERS FOR NN ####################
    # image parameters #
    imsize=32
    numfreqs = 1
    numclasses = 2 

    # layer parameters #
    w_init = None       # weight intializers for all layers
    n_layers = (4,2,1)  # num of convolutional layers in sequence
    poolsize = (2,2)    # maxpooling size
    filtersize = (3,3)  # filter size

    # fully connected output #
    size_fc = 512       # size of fully connected layers
    DROPOUT = True     # should we use Hinton Dropout method?

    # define number of epochs and batch size
    NUM_EPOCHS = 100 # per dataset
    batch_size = 24 # or 64... or 24
    data_augmentation = True

    ieegdnn = model.ieeg_cnn_rnn.IEEGdnn(imsize=imsize, 
                                        n_colors=numfreqs,
                                        num_classes=numclasses)
    sys.stdout.write('\n\n')
    sys.stdout.write(os.getcwd())

    ##################### TRAINING FOR NN ####################
    # VGG-12 style later
    currmodel = ieegdnn._build_2dcnn(w_init=w_init, n_layers=n_layers, 
                                  poolsize=poolsize, filter_size=filtersize)
    currmodel = ieegdnn._build_seq_output(currmodel, size_fc, DROPOUT)
    sys.stdout.write("Created VGG12 Style CNN")

    # VGG-12 style 3D CNN
    # poolsize = (2,2,2)    # maxpooling size
    # filtersize = (3,3,3)  # filter size
    # currmodel = ieegdnn._build_3dcnn(w_init=w_init, n_layers=n_layers, 
    #                               poolsize=poolsize, filter_size=filtersize)
    # currmodel = ieegdnn._build_seq_output(currmodel, size_fc, DROPOUT)
    # sys.stdout.write("Created VGG12 Style 3D CNN")

    print(currmodel.summary())
    if not os.path.exists(modeljsonfile):
        # serialize model to JSON
        model_json = currmodel.to_json()
        with open(modeljsonfile, "w") as json_file:
            json_file.write(model_json)
        print("Saved model to disk")

    # load in previous weights
    # bestweightsfile = 'final_weights.hdf5'
    # vggcnn.load_weights(bestweightsfile)
    # sys.stdout.write("Created model and loaded weights from file")

    # initialize loss function, SGD optimizer and metrics
    loss = 'binary_crossentropy'
    optimizer = keras.optimizers.Adam(lr=5e-4, 
                                    beta_1=0.9, 
                                    beta_2=0.999,
                                    epsilon=1e-08,
                                    decay=0.0)
    metrics = ['accuracy']

    modelconfig = currmodel.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    print(modelconfig)
    print("model input shape is: ", currmodel.input_shape)


    # This will do preprocessing and realtime data augmentation:
    datagen = keras.preprocessing.image.ImageDataGenerator(
                    featurewise_center=True,  # set input mean to 0 over the dataset
                    # samplewise_center=True,  # set each sample mean to 0
                    featurewise_std_normalization=True,  # divide inputs by std of the dataset
                    # samplewise_std_normalization=True,  # divide each input by its std
                    zca_whitening=False,      # apply ZCA whitening
                    # rotation_range=3,         # randomly rotate images in the range (degrees, 0 to 180)
                    # width_shift_range=0.02,    # randomly shift images horizontally (fraction of total width)
                    # height_shift_range=0.02,   # randomly shift images vertically (fraction of total height)
                    horizontal_flip=False,    # randomly flip images
                    vertical_flip=False,      # randomly flip images
                    # channel_shift_range=4,
                    fill_mode='nearest')  

    # checkpoint
    checkpoint = ModelCheckpoint(tempfilepath, 
                                    monitor='val_acc', 
                                    verbose=1, 
                                    save_best_only=True, 
                                    mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=4, min_lr=1e-6)
    callbacks = [checkpoint, reduce_lr] #, poly_decay]
    # INIT_LR = 5e-3
    G=1

    ##################### INPUT DATA FOR NN ####################
    data = np.load(alldatafile)
    images = data['image_tensor']

    # reshape
    images = images.reshape((-1, numfreqs, imsize, imsize))
    images = images.swapaxes(1,3)

    # load the ylabeled data
    ylabels = data['ylabels']
    invert_y = 1 - ylabels
    ylabels = np.concatenate((ylabels, invert_y),axis=1)

    # ERROR CHECK assert the shape of the images #
    sys.stdout.write("\n\n Images and ylabels shapes are: \n\n")
    print(images.shape)
    print(ylabels.shape)
    sys.stdout.write("\n\n") 
    assert images.shape[2] == images.shape[1]
    assert images.shape[2] == imsize
    assert images.shape[3] == numfreqs
    
    print(images.shape)
    images = images.astype("float32")
    # format the data correctly 
    X_train, X_test, y_train, y_test = train_test_split(images, ylabels, test_size=0.33, random_state=42)
   
    # augment data, or not and then trian the model!
    if not data_augmentation:
        print('Not using data augmentation. Implement Solution still!')
        HH = currmodel.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=NUM_EPOCHS,
                  validation_data=(X_test, y_test),
                  shuffle=False,
                  callbacks=callbacks)
    else:
        print('Using real-time data augmentation.')
        datagen.fit(X_train)
        HH = currmodel.fit_generator(
                    datagen.flow(X_train, y_train,batch_size=batch_size),
                            steps_per_epoch=images.shape[0] // batch_size,
                            epochs=NUM_EPOCHS,
                            validation_data=(X_test, y_test),
                            shuffle=True,
                            callbacks=callbacks, verbose=2)

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(images)

    with open(historyfile, 'wb') as file_pi:
        pickle.dump(HH.history, file_pi)
    # save final history object
    currmodel.save(finalweightsfile)

    # if running on test dataset of images
    predicted = currmodel.predict_classes(testimages)
    ytrue = np.argmax(testlabels, axis=1)

    y_pred = currmodel.predict(testimages)
    print("ROC_AUC_SCORES: ", roc_auc_score(testlabels, y_pred))
    # if running on validation dataset of images
    # predicted = currmodel.predict_classes(X_test)
    # ytrue = np.argmax(y_test, axis=1)

    print(ytrue.shape)
    print(predicted.shape)

    print('Mean accuracy score: ', accuracy_score(ytrue, predicted))
    print('F1 score:', f1_score(ytrue, predicted))
    print('Recall:', recall_score(ytrue, predicted))
    print('Precision:', precision_score(ytrue, predicted))
    print('\n clasification report:\n', classification_report(ytrue, predicted))
    print('\n confusion matrix:\n',confusion_matrix(ytrue, predicted))

