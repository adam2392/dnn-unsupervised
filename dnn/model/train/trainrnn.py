from .basetrain import BaseTrain
import numpy as np
import os
from sklearn.model_selection import train_test_split
# utilitiy libs
import ntpath
import json
import pickle

import keras
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator
import sklearn.utils
from .testingcallback import TestCallback

# metrics for postprocessing of the results
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, \
    recall_score, classification_report, \
    f1_score, roc_auc_score
import pprint

def preprocess_imgwithnoise(image_tensor):
    # preprocessing_function: function that will be implied on each input.
    #         The function will run before any other modification on it.
    #         The function should take one argument:
    #         one image (Numpy tensor with rank 3),
    #         and should output a Numpy tensor with the same shape.
    assert image_tensor.shape[0] == image_tensor.shape[1]
    stdmult = 0.1
    imsize = image_tensor.shape[0]
    numchans = image_tensor.shape[2]
    for i in range(numchans):
        feat = image_tensor[..., i]
        image_tensor[..., i] = image_tensor[..., i] + np.random.normal(
            scale=stdmult*np.std(feat), size=feat.size).reshape(imsize, imsize)
    return image_tensor

class TrainRNN(BaseTrain):
    def __init__(self, dnnmodel, batch_size, NUM_EPOCHS, AUGMENT):
        self.dnnmodel = dnnmodel
        self.batch_size = batch_size
        self.NUM_EPOCHS = NUM_EPOCHS
        self.AUGMENT = AUGMENT

    def summaryinfo(self):
        summary = {
            'batch_size': self.batch_size,
            'epochs': self.NUM_EPOCHS,
            'augment': self.AUGMENT,
            'class_weight': self.class_weight
        }
        pprint.pprint(summary)

    def configure(self, tempdatadir):
        # initialize loss function, SGD optimizer and metrics
        loss = 'binary_crossentropy'
        optimizer = RMSprop(lr=1e-4,
                            rho=0.9,
                            epsilon=1e-08,
                            decay=0.0)
        metrics = ['accuracy']
        self.modelconfig = self.dnnmodel.compile(loss=loss,
                                                 optimizer=optimizer,
                                                 metrics=metrics)

        tempfilepath = os.path.join(
            tempdatadir, "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5")
        # callbacks availabble
        checkpoint = ModelCheckpoint(tempfilepath,
                                     monitor='val_acc',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='max')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                      patience=10, min_lr=1e-8)
        testcheck = TestCallback()
        self.callbacks = [checkpoint, reduce_lr, testcheck]

    def _formatdata(self, images):
        images = images.swapaxes(1, 3)
        # lower sample by casting to 32 bits
        images = images.astype("float32")
        return images

    def train(self):
        self._loadgenerator()

        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test
        class_weight = self.class_weight
        callbacks = self.callbacks
        dnnmodel = self.dnnmodel

        print("Training data: ", X_train.shape, y_train.shape)
        print("Testing data: ", X_test.shape, y_test.shape)
        print("Class weights are: ", class_weight)

        test = np.argmax(y_train, axis=1)
        print("class imbalance: ", np.sum(test), len(test))

        # augment data, or not and then trian the model!
        if not self.AUGMENT:
            print('Not using data augmentation. Implement Solution still!')
            HH = dnnmodel.fit(X_train, y_train,
                              # steps_per_epoch=X_train.shape[0] // self.batch_size,
                              batch_size = self.batch_size,
                              epochs=self.NUM_EPOCHS,
                              validation_data=(X_test, y_test),
                              shuffle=True,
                              class_weight=class_weight,
                              callbacks=callbacks)
        else:
            print('Using real-time data augmentation.')
            # self.generator.fit(X_train)
            HH = dnnmodel.fit_generator(self.generator.flow(X_train, y_train, batch_size=self.batch_size),
                                        steps_per_epoch=X_train.shape[0] // self.batch_size,
                                        epochs=self.NUM_EPOCHS,
                                        validation_data=(X_test, y_test),
                                        shuffle=True,
                                        class_weight=class_weight,
                                        callbacks=callbacks, verbose=2)

        self.HH = HH
    
    def _loadgenerator(self):
        # This will do preprocessing and realtime data augmentation:
        self.generator = ImageDataGenerator(
            # featurewise_cente/r=True,  # set input mean to 0 over the dataset
            # samplewise_center=True,  # set each sample mean to 0
            # featurewise_std_normalization=True,  # divide inputs by std of the dataset
            # samplewise_std_normalization=True,  # divide each input by its std
            # zca_whitening=False,      # apply ZCA whitening
            # randomly rotate images in the range (degrees, 0 to 180)
            # rotation_range=5,
            # randomly shift images horizontally (fraction of total width)
            # width_shift_range=0.2,
            # randomly shift images vertically (fraction of total height)
            # height_shift_range=0.2,
            # horizontal_flip=True,    # randomly flip images
            # vertical_flip=True,      # randomly flip images
            # channel_shift_range=4,
            # fill_mode='nearest',
            preprocessing_function=preprocess_imgwithnoise)
