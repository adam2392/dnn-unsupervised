from .basetrain import BaseTrain
import numpy as np
import os
from sklearn.model_selection import train_test_split

import keras
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from .testingcallback import TestCallback
from keras.preprocessing.image import ImageDataGenerator

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


class TrainFragAux(BaseTrain):
    def __init__(self, dnnmodel, batch_size, NUM_EPOCHS, AUGMENT):
        self.dnnmodel = dnnmodel
        self.batch_size = batch_size
        self.NUM_EPOCHS = NUM_EPOCHS
        self.AUGMENT = AUGMENT

    def configure(self, tempdatadir):
        # initialize loss function, SGD optimizer and metrics
        loss = 'binary_crossentropy'
        optimizer = Adam(lr=1e-4,
                         beta_1=0.9,
                         beta_2=0.999,
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

    def loadformatteddata(self, Xmain_train, Xmain_test,
                          Xaux_train, Xaux_test, y_train, y_test, class_weight):
        self.Xmain_train = Xmain_train
        self.Xmain_test = Xmain_test
        self.Xaux_train = Xaux_train
        self.Xaux_test = Xaux_test
        self.y_train = y_train
        self.y_test = y_test
        self.class_weight = class_weight

    def train(self):
        # get the training parameters
        callbacks = self.callbacks
        dnnmodel = self.dnnmodel
        class_weight = self.class_weight

        # Finally create generator
        gen_flow = self.gen_flow_for_two_inputs(
            self.Xmain_train, self.Xaux_train, self.y_train)

        print("main data shape: ", self.Xmain_train.shape)
        print("aux data shape: ", self.Xaux_train.shape)
        print("y data shape: ", self.y_train.shape)

        # augment data, or not and then trian the model!
        if not self.AUGMENT:
            print('Not using data augmentation. Implement Solution still!')
            # HH = dnnmodel.fit(X_train, y_train,
            #             steps_per_epoch=X_train.shape[0] // batch_size,
            #             batch_size=self.batch_size,
            #             epochs=self.NUM_EPOCHS,
            #             validation_data=(X_test, y_test),
            #             shuffle=True,
            #             class_weight=class_weight,
            #             callbacks=callbacks)
        else:
            print('Using real-time data augmentation.')
            self.generator.fit(self.Xaux_train)
            HH = dnnmodel.fit_generator(gen_flow,
                                        steps_per_epoch=self.y_train.shape[0] // self.batch_size,
                                        epochs=self.NUM_EPOCHS,
                                        validation_data=(
                                            self.Xmain_test, self.Xaux_test, self.y_test),
                                        shuffle=True,
                                        class_weight=class_weight,
                                        callbacks=callbacks, verbose=2)
        self.HH = HH

    def loadgenerator(self):
        # This will do preprocessing and realtime data augmentation:
        self.generator = ImageDataGenerator(
            # featurewise_center=True,  # set input mean to 0 over the dataset
            samplewise_center=True,  # set each sample mean to 0
            # featurewise_std_normalization=True,  # divide inputs by std of the dataset
            samplewise_std_normalization=True,  # divide each input by its std
            zca_whitening=False,      # apply ZCA whitening
            # randomly rotate images in the range (degrees, 0 to 180)
            rotation_range=5,
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.2,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.2,
            horizontal_flip=False,    # randomly flip images
            vertical_flip=False,      # randomly flip images
            channel_shift_range=4,
            fill_mode='nearest',
            preprocessing_function=preprocess_imgwithnoise)

    # Here is the function that merges our two generators
    # We use the exact same generator with the same random seed for both the y and angle arrays
    def gen_flow_for_two_inputs(self, X1, X2, y):
        genX1 = self.generator.flow(
            X1, y, batch_size=self.batch_size, seed=666)
        genX2 = self.generator.flow(
            X2, y, batch_size=self.batch_size, seed=666)

        while True:
            X1i = genX1.next()
            X2i = genX2.next()
            # Assert arrays are equal - this was for peace of mind, but slows down training
            # np.testing.assert_array_equal(X1i[0],X2i[0])
            yield [X1i[0], X2i[0]], X1i[1]

    def saveoutput(self, modelname, outputdatadir):
        modeljsonfile = os.path.join(outputdatadir, modelname + "_model.json")
        historyfile = os.path.join(
            outputdatadir,  modelname + '_history' + '.pkl')
        finalweightsfile = os.path.join(
            outputdatadir, modelname + '_final_weights' + '.h5')

        # save model
        if not os.path.exists(modeljsonfile):
            # serialize model to JSON
            model_json = currmodel.to_json()
            with open(modeljsonfile, "w") as json_file:
                json_file.write(model_json)
            print("Saved model to disk")

        # save history
        with open(historyfile, 'wb') as file_pi:
            pickle.dump(self.HH.history, file_pi)

        # save final weights
        self.dnnmodel.save(finalweightsfile)
