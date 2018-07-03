import os
import numpy as np
import keras
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

import dnn.base.constants.model_constants as MODEL_CONSTANTS
from dnn.keras_models.trainers.base import BaseTrainer
from dnn.keras_models.trainers.callbacks.testingcallback import MetricsCallback
from dnn.util.keras.augmentations import Augmentations 

from keras.metrics import categorical_accuracy

from dnn.keras_models.metrics.classifier import BinaryClassifierMetric
from dnn.base.constants.config import Config, OutputConfig
from dnn.keras_models.regularizer.post_class_regularizer import Postalarm
# import tensorboardX  # import SummaryWriter
# from tqdm import trange

class CNNTrainer(BaseTrainer):
    metric_comp = BinaryClassifierMetric()
    post_regularizer = None
    HH = None

    def __init__(self, model, num_epochs=MODEL_CONSTANTS.NUM_EPOCHS, 
                 batch_size=MODEL_CONSTANTS.BATCH_SIZE,
                 outputdir=None,
                 learning_rate=MODEL_CONSTANTS.LEARNING_RATE,
                 shuffle=MODEL_CONSTANTS.SHUFFLE,
                 augment=MODEL_CONSTANTS.AUGMENT,
                 config=None):
        '''         SET LOGGING DIRECTORIES: MODEL, TENSORBOARD         '''
        self.outputdir = outputdir
        super(CNNTrainer, self).__init__(model=model,
                                         config=config)

        # Hyper parameters - training
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        # hyper parameters - dataset
        self.shuffle = shuffle
        self.AUGMENT = augment

        self.save_summary_steps = 10
        self.gradclip_value = 1.5

        # set tensorboard writer
        self._setdirs()  # set directories for all logging
        # self.writer = tensorboardX.SummaryWriter(self.tboardlogdir)

        self.logger.info(
            "Logging output data to: {}".format(
                self.outputdatadir))
        self.logger.info(
            "Logging experimental data at: {}".format(
                self.explogdir))
        self.logger.info(
            "Logging tensorboard data at: {}".format(
                self.tboardlogdir))

    def _setdirs(self):
        # set where to log outputs of explog
        if self.outputdir is None:
            self.explogdir = os.path.join(
                self.config.tboard.FOLDER_LOGS, 'traininglogs')
            self.tboardlogdir = os.path.join(
                self.config.tboard.FOLDER_LOGS, 'tensorboard')
            self.outputdatadir = os.path.join(
                self.config.tboard.FOLDER_LOGS, 'output')
        else:
            self.explogdir = os.path.join(self.outputdir, 'traininglogs')
            self.tboardlogdir = os.path.join(self.outputdir, 'tensorboard')
            self.outputdatadir = os.path.join(self.outputdir, 'output')

        if not os.path.exists(self.explogdir):
            os.makedirs(self.explogdir)
        if not os.path.exists(self.tboardlogdir):
            os.makedirs(self.tboardlogdir)
        if not os.path.exists(self.outputdatadir):
            os.makedirs(self.outputdatadir)

    def saveoutput(self, modelname):
        modeljson_filepath = os.path.join(self.outputdatadir, modelname + "_model.json")
        history_filepath = os.path.join(
            self.outputdatadir,  modelname + '_history' + '.pkl')
        finalweights_filepath = os.path.join(
            self.outputdatadir, modelname + '_final_weights' + '.h5')
        self._saveoutput(modeljson_filepath, history_filepath, finalweights_filepath)

    def savemetricsoutput(self, modelname):
        metricsfilepath = os.path.join(self.outputdatadir, modelname+ "_metrics.json")
        auc = self.metrichistory.aucs
        fpr = self.metrichistory.fpr
        tpr = self.metrichistory.tpr 
        thresholds = self.metrichistory.thresholds
        metricdata = {
            'auc': auc,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
        }
        self._writejsonfile(metricdata, metricsfilepath)

    def composedatasets(self, train_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        # get input characteristics
        self.imsize = train_dataset.imsize
        self.n_colors = train_dataset.n_colors
        # size of training/testing set
        self.train_size = len(train_dataset)
        self.val_size = len(test_dataset)
        self.steps_per_epoch = self.train_size // self.batch_size

        # self.train_size = 2#len(train_dataset_obj)
        # self.val_size = 2#len(test_dataset_obj)
        self.logger.info(
            "Each training epoch is {} steps and each validation is {} steps.".format(
                self.train_size, self.val_size))
        self.logger.info(
            "Setting the datasets for training/testing in trainer object!")
        self.logger.info(
            "Image size is {} with {} colors".format(
                self.imsize, self.n_colors))

    def configure(self):
        """
        Configuration function that can change:
        - sets optimizer
        - sets loss function
        - sets scheduler
        - sets post-prediction-regularizer
        """
        # initialize loss function, SGD optimizer and metrics
        clipnorm = 1.
        model_params = {
            'loss': 'binary_crossentropy',
            'optimizer': Adam(beta_1=0.9,
                         beta_2=0.99,
                         epsilon=1e-08,
                         decay=0.0,
                         amsgrad=True,
                         clipnorm=clipnorm),
            'metrics': [categorical_accuracy]
        }
        self.modelconfig = self.model.compile(**model_params)

        tempfilepath = os.path.join(self.explogdir, "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5")

        '''                         CREATE CALLBACKS                        '''
        # callbacks availabble
        checkpoint = ModelCheckpoint(tempfilepath,
                                     monitor=categorical_accuracy,
                                     verbose=1,
                                     save_best_only=True,
                                     mode='max')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                    factor=0.5,
                                    patience=10, 
                                    min_lr=1e-8)
        tboard = keras.callbacks.TensorBoard(log_dir=self.tboardlogdir, 
                                    histogram_freq=self.num_epochs/5, 
                                    batch_size=self.batch_size, write_graph=True, 
                                    write_grads=True, 
                                    # write_images=True, 
                                    embeddings_freq=0, 
                                    embeddings_layer_names=None, 
                                    embeddings_metadata=None, 
                                    embeddings_data=None)
        metrichistory = MetricsCallback()
        self.callbacks = [checkpoint,
                        reduce_lr,
                        tboard,
                        metrichistory]

    def train(self):
        self._loadgenerator()
        print("Training data: ", self.train_dataset.X_train.shape,  self.train_dataset.y_train.shape)
        print("Testing data: ",  self.test_dataset.X_test.shape,  self.test_dataset.y_test.shape)
        print("Class weights are: ",  self.train_dataset.class_weight)
        test = np.argmax( self.train_dataset.y_train, axis=1)
        print("class imbalance: ", np.sum(test), len(test))

        # augment data, or not and then trian the model!
        if not self.AUGMENT:
            print('Not using data augmentation. Implement Solution still!')
            HH = self.model.fit( self.train_dataset.X_train,  self.train_dataset.y_train,
                              # steps_per_epoch=X_train.shape[0] // self.batch_size,
                              batch_size = self.batch_size,
                              epochs=self.NUM_EPOCHS,
                              validation_data=(self.test_dataset.X_test, self.test_dataset.y_test),
                              shuffle=True,
                              class_weight= self.train_dataset.class_weight,
                              callbacks=self.callbacks)
        elif self.AUGMENT=='dir':
            directory = self.train_directory
            target_size = (self.length_imsize, self.width_imsize)
            color_mode = 'grayscale'
            classes=[0, 1]
            
            HH = self.model.fit_generator(self.generator.flow_from_directory(self, directory,
                                                            target_size=target_size, color_mode=color_mode,
                                                            classes=None, 
                                                            class_mode='categorical',
                                                            batch_size=self.batch_size, 
                                                            shuffle=self.shuffle, 
                                                            interpolation='nearest'),
                                        steps_per_epoch=self.steps_per_epoch,
                                        epochs=self.num_epochs,
                                        validation_data=(self.test_dataset.X_test, self.test_dataset.y_test),
                                        shuffle=self.shuffle,
                                        class_weight= self.train_dataset.class_weight,
                                        callbacks=self.callbacks, verbose=2)
        else:
            print('Using real-time data augmentation.')
            # self.generator.fit(X_train)
            HH = self.model.fit_generator(self.generator.flow(self.train_dataset.X_train, self.train_dataset.y_train, 
                                                                batch_size=self.batch_size),
                                        steps_per_epoch=self.steps_per_epoch,
                                        epochs=self.num_epochs,
                                        validation_data=(self.test_dataset.X_test, self.test_dataset.y_test),
                                        shuffle=True,
                                        class_weight= self.train_dataset.class_weight,
                                        callbacks=self.callbacks, verbose=2)

        self.HH = HH
        self.metrichistory = self.callbacks[3] 

    def _loadgenerator(self):
        imagedatagen_args = {
            'featurewise_center':True,  # set input mean to 0 over the dataset
            'samplewise_center':False,  # set each sample mean to 0
            'featurewise_std_normalization':True,  # divide inputs by std of the dataset
            'samplewise_std_normalization':False,  # divide each input by its std
            'zca_whitening':False,      # apply ZCA whitening
            # randomly rotate images in the range (degrees, 0 to 180)
            'rotation_range':5,
            # randomly shift images horizontally (fraction of total width)
            'width_shift_range':0.2,
            # randomly shift images vertically (fraction of total height)
            'height_shift_range':0.2,
            'horizontal_flip':True,    # randomly flip images
            'vertical_flip':True,      # randomly flip images
            'channel_shift_range':4,
            'fill_mode':'nearest',
            'preprocessing_function':Augmentations.preprocess_imgwithnoise
        }

        # This will do preprocessing and realtime data augmentation:
        self.generator = ImageDataGenerator(**imagedatagen_args)

if __name__ == '__main__':
    from dnn.keras_models.nets.cnn import iEEGCNN
    from dnn.io.readerimgdataset import ReaderImgDataset 
    import dnn.base.constants.model_constants as constants

    data_procedure = 'loo'
    testpat = 'id001_bt'
    traindir = os.path.expanduser('~/Downloads/tngpipeline/freq/fft_img/')
    testdir = traindir
    # initialize reader to get the training/testing data
    reader = ReaderImgDataset()
    reader.loadbydir(traindir, testdir, procedure=data_procedure, testname=testpat)
    reader.loadfiles(mode=constants.TRAIN)
    reader.loadfiles(mode=constants.TEST)
    
    # create the dataset objects
    train_dataset = reader.train_dataset
    test_dataset = reader.test_dataset

    # define model
    model_params = {
        'num_classes': 2,
        'imsize': 64,
        'n_colors':4,
    }
    model = iEEGCNN(**model_params) 
    model.buildmodel(output=True)

    num_epochs = 1
    batch_size = 32
    outputdir = './'
    trainer = CNNTrainer(model=model.net, num_epochs=num_epochs, 
                        batch_size=batch_size,
                        outputdir=outputdir)
    trainer.composedatasets(train_dataset, test_dataset)
    trainer.configure()
    # Train the model
    # trainer.train()
    modelname='test'
    trainer.saveoutput(modelname=modelname)
    trainer.savemetricsoutput(modelname=modelname)
    print(model.net)
