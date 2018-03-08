from .basetrain import BaseTrain 
import numpy as np
import os

# keras functions
import keras
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import Callback

from sklearn.utils import class_weight
import sys
sys.path.append('../../')

# data dir generator
from processing.generators import genseqfromfile
from sklearn.model_selection import train_test_split

# metrics for postprocessing of the results
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, \
    recall_score, classification_report, \
    f1_score, roc_auc_score

# for smart splitting of lists
import more_itertools as mit
import pprint

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

class TestCallback(Callback):
    def __init__(self):
        # self.test_data = test_data
        self.aucs = []

    def on_epoch_end(self, epoch, logs={}):
        # x, y = self.test_data
        # x = self.model.validation_data[0]
        # y = self.model.validation_data[1]
        x = self.validation_data[0]
        y = self.validation_data[1]

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

class TrainSeq(BaseTrain):
    def __init__(self, dnnmodel, batch_size, numtimesteps, NUM_EPOCHS, AUGMENT):
        self.dnnmodel = dnnmodel                # the dnn model we will use to train
        self.batch_size = batch_size            # the batch size per training epoch
        self.numtimesteps = numtimesteps        # the number of time steps in our sequence data
        self.NUM_EPOCHS = NUM_EPOCHS            # epochs to train on
        self.AUGMENT = AUGMENT                  # augment data or not?


        self.imsize = None
        self.numchans = None

    def summaryinfo(self):
        summary = {
            'batch_size': self.batch_size,
            'epochs': self.NUM_EPOCHS,
            'augment': self.AUGMENT,
            'numtimesteps': self.numtimesteps
        }
        pprint.pprint(summary)

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
        optimizer = RMSprop(lr=1e-4, 
                        rho=0.9, 
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
        self.callbacks = [checkpoint, reduce_lr, testcheck]

    def loadmodel(self, modelfile, weightsfile):
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

    # To Do: LOAD DATA FOR SEQUENCEs now
    def loaddata(self, datafile, imsize, numchans):
        pass

    def formatdata(self, seqofimgs, seqoflabels):
        '''
        To Do Formatting:
        1. Stateful training: this means that each training depends on the previous.
            Can try batch_size == 1, if so, or not...
            
            However, training samples need to be in order for lstm to make sense of data
            from previous training.... -> train per patient
        2. Many to many: Each Sequence of LSTM predicts all the windows as "seizure or not"

        3. Many to one: Each sequence of LSTM predicts the window of interest (last one, or middle one),
                depending on bidirectional or not and then says "seizure or not"

        '''
        numsamps, imsize, _, numchans = seqofimgs.shape
        assert seqofimgs.ndim == 4 # (samples, imsize, imsize, channel)
        assert imsize == self.imsize
        assert numchans == self.numchans

        # what is our desired shape for the sequence data
        desired_shape = (self.numtimesteps, self.imsize, self.imsize, self.numchans)
        img_shape = (self.imsize, self.imsize, self.numchans)

        formatted_X = []
        formatted_Y = []

        # Option 1: use numpy array split to split along the channels
        formatted_X = list(mit.windowed(seqofimgs, 
                                        n=self.numtimesteps, 
                                        step=self.numtimesteps//2))
        formatted_Y = list(mit.windowed(seqoflabbels, 
                                        n=self.numtimesteps, 
                                        step=self.numtimesteps//2))

        # use list comprehension to supposedly get the same answer
        size = self.numtimesteps
        step = self.numtimesteps //2
        formatted_X = [seqofimgs[i : i + size, ...] for i in range(0, numsamps, step)]
        formatted_Y = [seqoflabels[i : i + size, ...] for i in range(0, numsamps, step)]

        # Option 2: randomly choose index 
        randind = np.random.choice(numsamps)
        part_data = seqofimgs[randind:randind+self.numtimesteps, ...]
        part_y = seqoflabbels[randind:randind+self.numtimesteps]
        formatted_X.append(part_data)
        formatted_Y.append(part_y)

        # pad beginning and end with zeros if necessary that is the shape of an img
        formatted_X = self._datapad(formatted_X, padding='post')
        formatted_Y = self._datapad(formatted_Y, padding='post')

    def _paddata(self, sampdata, padding):
        '''
        Pad a sample data on the 
        '''
        samplen = sampdata.shape[0]
        numsamps_topad = self.num_timesteps - samplen
        imshape = sampdata.shape[1:]
        assert imshape[0] == self.imsize
        assert imshape[-1] == self.numchannels

        # pad with zeros
        if samplen < self.num_timesteps:
            padmat = np.zeros(((samplen,)+imshape))
            if padding == 'pre':
                # pad on the pre
                sampdata = np.concatenate((padmat, sampdata), axis=0)
            elif padding == 'post':
                # pad on the pre
                sampdata = np.concatenate((sampdata, padmat), axis=0)
        return sampdata

    def loaddirofdata(self, datadir, listofpats):
        ''' Get list of file paths '''
        self.filepaths = []
        self.testfilepaths = []
        for root, dirs, files in os.walk(datadir):
            for file in files:
                if any(pat in file for pat in listofpats):
                    self.filepaths.append(os.path.join(root, file))
                else:
                    self.testfilepaths.append(os.path.join(root,file))
        self.samples = len(self.filepaths)

    def compute_classweights(self):
        '''     LOAD TRAINING DATA -> COMPUTE CLASS WEIGHTS     '''
        for idx, datafile in enumerate(self.filepaths):                    
            imagedata = np.load(datafile)
            metadata = imagedata['metadata'].item()
            if idx == 0:
                ylabels = metadata['ylabels']
            else:
                ylabels = np.append(ylabels, metadata['ylabels'], axis=0)

        # load the ylabeled data 1 in 0th position is 0, 1 in 1st position is 1
        invert_y = 1 - ylabels
        ylabels = np.concatenate((invert_y, ylabels),axis=1)  
        # format the data correctly 
        class_weight = sklearn.utils.compute_class_weight('balanced', 
                                                 np.unique(ylabels).astype(int),
                                                 np.argmax(ylabels, axis=1))
        self.class_weight = class_weight

    def train(self):
        class_weight = self.class_weight
        callbacks = self.callbacks
        dnnmodel = self.dnnmodel
        num_timesteps = self.num_timesteps

        print('Using real-time data augmentation.')
        HH = dnnmodel.fit_generator(self.generator.flow(X_train, y_train, batch_size=self.batch_size),
                                    num_timesteps=num_timesteps,
                                    steps_per_epoch=X_train.shape[0] // self.batch_size,
                                    epochs=self.NUM_EPOCHS,
                                    validation_data=(X_test, y_test),
                                    shuffle=True,
                                    class_weight=class_weight,
                                    callbacks=callbacks, verbose=2)

        self.HH = HH

    def loadgenerator(self):
        # This will do preprocessing and realtime data augmentation:
        self.generator = imagegen = genseqfromfile.DataDirGenerator(
                                featurewise_center=False,  # set input mean to 0 over the dataset
                                samplewise_center=True,  # set each sample mean to 0
                                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                samplewise_std_normalization=True,  # divide each input by its std
                                rotation_range=5,         # randomly rotate images in the range (degrees, 0 to 180)
                                width_shift_range=0.02,    # randomly shift images horizontally (fraction of total width)
                                height_shift_range=0.02,   # randomly shift images vertically (fraction of total height)
                                horizontal_flip=False,    # randomly flip images
                                vertical_flip=False,      # randomly flip images
                                shear_range=0.,
                                zoom_range=0.,
                                channel_shift_range=4.,
                                fill_mode='nearest',
                                preprocessing_function=preprocess_imgwithnoise) 



