from .basetrain import BaseTrain 
import numpy as np
import os
from sklearn.model_selection import train_test_split
# utilitiy libs
import ntpath
import json
import pickle

import keras
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import Callback

from sklearn.utils import class_weight

# metrics for postprocessing of the results
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, \
    recall_score, classification_report, \
    f1_score, roc_auc_score

def preprocess_imgwithnoise(image_tensor):
    # preprocessing_function: function that will be implied on each input.
    #         The function will run before any other modification on it.
    #         The function should take one argument:
    #         one image (Numpy tensor with rank 3),
    #         and should output a Numpy tensor with the same shape.
    assert image_tensor.shape[0] == image_tensor.shape[1]
    stdmult=0.1
    imsize = image_tensor.shape[0]
    numchans = image_tensor.shape[2]
    for i in range(numchans):
        feat = image_tensor[...,i]
        image_tensor[...,i] = image_tensor[...,i] + np.random.normal(scale=stdmult*np.std(feat), size=feat.size).reshape(imsize,imsize)
    return image_tensor
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
class TrainCNN(BaseTrain):
    def __init__(self, dnnmodel, batch_size, NUM_EPOCHS, AUGMENT):
        self.dnnmodel = dnnmodel
        self.batch_size = batch_size
        self.NUM_EPOCHS = NUM_EPOCHS
        self.AUGMENT = AUGMENT

    def saveoutput(self, modelname, outputdatadir):
        modeljsonfile = os.path.join(outputdatadir, modelname + "_model.json")
        historyfile = os.path.join(outputdatadir,  modelname + '_history'+ '.pkl')
        finalweightsfile = os.path.join(outputdatadir, modelname + '_final_weights' + '.h5')

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

    def testoutput(self):
        y_test = self.y_test

        prob_predicted = self.dnnmodel.predict(self.X_test)
        ytrue = np.argmax(y_test, axis=1)
        y_pred = currmodel.predict_classes(self.X_test)

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

        tempfilepath = os.path.join(tempdatadir,"weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5")
        
        # callbacks availabble
        checkpoint = ModelCheckpoint(tempfilepath, 
                                    monitor='val_acc', 
                                    verbose=1, 
                                    save_best_only=True, 
                                    mode='max')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=10, min_lr=1e-8)
        testcheck = TestCallback()
        callbacks = [checkpoint, reduce_lr, testcheck]

    def loaddata(self, datafile, imsize, numchans):
        ''' Get all data '''
        ##################### INPUT DATA FOR NN ####################
        data = np.load(alldatafile)
        images = data['image_tensor']   # load image
        ylabels = data['ylabels']       # load ylabels

        # reshape
        images = images.reshape((-1, numchans, imsize, imsize))
        images = images.swapaxes(1,3)

        # load the ylabeled data 1 in 0th position is 0, 1 in 1st position is 1
        invert_y = 1 - ylabels
        ylabels = np.concatenate((invert_y, ylabels),axis=1)

        # ERROR CHECK assert the shape of the images #
        sys.stdout.write("\n\n Images and ylabels shapes are: \n\n")
        sys.stdout.write(images.shape)
        sys.stdout.write(ylabels.shape)
        sys.stdout.write("\n\n") 

        assert images.shape[2] == images.shape[1]
        assert images.shape[2] == imsize
        assert images.shape[3] == numfreqs

        # lower sample by casting to 32 bits
        images = images.astype("float32")

        self.images = images
        self.ylabels = ylabels

        # format the data correctly 
        X_train, X_test, y_train, y_test = train_test_split(images, ylabels, test_size=0.33, random_state=42)
    
        class_weight = class_weight.compute_class_weight('balanced', 
                                                 np.unique(ylabels).astype(int),
                                                 np.argmax(ylabels, axis=1))

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.class_weight = class_weight

    def loaddirofdata(self, datadir, listofpats, LOAD):
        ''' Get list of file paths '''
        self.filepaths = []
        self.testfilepaths = []
        for root, dirs, files in os.walk(self.directory):
            for file in files:
                if any(pat in file for pat in listofpats)
                    self.filepaths.append(os.path.join(root, file))
                else:
                    self.testfilepaths.append(os.path.join(root,file))
        self.samples = len(self.filepaths)

        if LOAD:
            '''     LOAD TRAINING DATA      '''
            for idx, datafile in enumerate(self.filepaths):                    
                imagedata = np.load(datafile)
                image_tensor = imagedata['image_tensor']
                metadata = imagedata['metadata'].item()
                
                if idx == 0:
                    image_tensors = image_tensor
                    ylabels = metadata['ylabels']
                else:
                    image_tensors = np.append(image_tensors, image_tensor, axis=0)
                    ylabels = np.append(ylabels, metadata['ylabels'], axis=0)

            # load the ylabeled data 1 in 0th position is 0, 1 in 1st position is 1
            invert_y = 1 - ylabels
            ylabels = np.concatenate((invert_y, ylabels),axis=1)  
            # format the data correctly 
            class_weight = class_weight.compute_class_weight('balanced', 
                                                     np.unique(ylabels).astype(int),
                                                     np.argmax(ylabels, axis=1))
            self.X_train = image_tensor
            self.y_train = ylabels

            '''     LOAD TESTING DATA      '''
            for idx, datafile in enumerate(self.testfilepaths):
                imagedata = np.load(datafile)
                image_tensor = imagedata['image_tensor']
                metadata = imagedata['metadata'].item()
                
                if idx == 0:
                    image_tensors = image_tensor
                    ylabels = metadata['ylabels']
                else:
                    image_tensors = np.append(image_tensors, image_tensor, axis=0)
                    ylabels = np.append(ylabels, metadata['ylabels'], axis=0)
            # load the ylabeled data 1 in 0th position is 0, 1 in 1st position is 1
            invert_y = 1 - ylabels
            ylabels = np.concatenate((invert_y, ylabels),axis=1)  

            self.X_test = image_tensors
            self.y_test = ylabels
            self.class_weight = class_weight

    def train(self):
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test
        class_weight = self.class_weight
        callbacks = self.callbacks
        dnnmodel = self.dnnmodel

        # augment data, or not and then trian the model!
        if not self.AUGMENT:
            print('Not using data augmentation. Implement Solution still!')
            HH = dnnmodel.fit(X_train, y_train,
                                steps_per_epoch=X_train.shape[0] // batch_size,
                                batch_size=self.batch_size,
                                epochs=self.NUM_EPOCHS,
                                validation_data=(X_test, y_test),
                                shuffle=True,
                                class_weight=class_weight,
                                callbacks=callbacks)
        else:
            print('Using real-time data augmentation.')
            self.generator.fit(X_train)
            HH = dnnmodel.fit_generator(self.generator.flow(X_train, y_train, batch_size=self.batch_size),
                                                steps_per_epoch=X_train.shape[0] // self.batch_size,
                                                epochs=self.NUM_EPOCHS,
                                                validation_data=(X_test, y_test),
                                                shuffle=True,
                                                class_weight=class_weight,
                                                callbacks=callbacks, verbose=2)

        self.HH = HH

    def loadgenerator(self):
        # This will do preprocessing and realtime data augmentation:
        self.generator = keras.preprocessing.image.ImageDataGenerator(
                    featurewise_center=True,  # set input mean to 0 over the dataset
                    # samplewise_center=True,  # set each sample mean to 0
                    featurewise_std_normalization=True,  # divide inputs by std of the dataset
                    # samplewise_std_normalization=True,  # divide each input by its std
                    zca_whitening=False,      # apply ZCA whitening
                    rotation_range=5,         # randomly rotate images in the range (degrees, 0 to 180)
                    width_shift_range=0.2,    # randomly shift images horizontally (fraction of total width)
                    height_shift_range=0.2,   # randomly shift images vertically (fraction of total height)
                    horizontal_flip=False,    # randomly flip images
                    vertical_flip=False,      # randomly flip images
                    channel_shift_range=4,
                    fill_mode='nearest',
                    preprocessing_function=preprocess_imgwithnoise)  
