import os
import numpy as np
import keras
from keras.optimizers import Adam
import keras.backend as K

import dnn.base.constants.model_constants as MODEL_CONSTANTS
from dnn.keras_models.trainers.base import BaseTrainer

# import callbacks and augmentation functions
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from dnn.keras_models.trainers.callbacks.testingcallback import MetricsCallback
from dnn.util.keras.augmentations import Augmentations 

# import generator(s) for loading in data
from dnn.util.generators.auxseq.generator import AuxImgDataGenerator

from keras.metrics import categorical_accuracy

# import metrics for postprocessing - metric analysis
from dnn.keras_models.metrics.classifier import BinaryClassifierMetric
from dnn.keras_models.regularizer.post_class_regularizer import Postalarm

class EZNetTrainer(BaseTrainer):
    metric_comp = BinaryClassifierMetric()
    post_regularizer = None
    HH = None

    def __init__(self, model, 
                num_epochs=MODEL_CONSTANTS.NUM_EPOCHS, 
                 batch_size=MODEL_CONSTANTS.BATCH_SIZE,
                 outputdir=None,
                 learning_rate=MODEL_CONSTANTS.LEARNING_RATE,
                 shuffle=MODEL_CONSTANTS.SHUFFLE,
                 augment=MODEL_CONSTANTS.AUGMENT,
                 config=None):
        '''         SET LOGGING DIRECTORIES: MODEL, TENSORBOARD         '''
        self.outputdir = outputdir
        super(EZNetTrainer, self).__init__(model=model,
                                         config=config)

        # Hyper parameters - training
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        # hyper parameters - dataset
        self.shuffle = shuffle
        self.AUGMENT = augment

        self.save_summary_steps = 10
        self.gradclip_value = 2.0

        # set tensorboard writer
        self._setdirs()  # set directories for all logging
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
        self.imsize = (train_dataset.length_imsize, train_dataset.width_imsize)
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
        from itertools import product
        import functools
        def w_categorical_crossentropy(y_true, y_pred, weights):
            ''' https://github.com/keras-team/keras/issues/2115
                https://stackoverflow.com/questions/46202839/weight-different-misclassifications-differently-keras

             '''
            weights = np.array(weights)
            nb_cl = weights.shape[0]
            final_mask = K.zeros_like(y_pred[:, 0])
            y_pred_max = K.max(y_pred, axis=1)
            y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
            y_pred_max_mat = K.equal(y_pred, y_pred_max)
            
            final_mask += (weights * y_pred_max_mat[:, :] * y_true[:, :])
            return K.categorical_crossentropy(y_pred, y_true) * final_mask
        from keras.losses import binary_crossentropy
        def weighted_binary_crossentropy(y_true, y_pred):
            false_positive_weight = self.train_dataset.class_weight[1]        
            # false_negative_weight = self.train_dataset.class_weight[1]
            thresh = 0.5
            y_pred_true = K.greater_equal(thresh,y_pred)
            y_not_true = K.less_equal(thresh,y_true)
            false_positive_tensor = K.equal(y_pred_true,y_not_true)

            #changing from here

            #first let's transform the bool tensor in numbers - maybe you need float64 depending on your configuration
            false_positive_tensor = K.cast(false_positive_tensor,'float32') 

            #and let's create it's complement (the non false positives)
            complement = 1 - false_positive_tensor

            #now we're going to separate two groups
            falsePosGroupTrue = y_true * false_positive_tensor
            falsePosGroupPred = y_pred * false_positive_tensor

            nonFalseGroupTrue = y_true * complement
            nonFalseGroupPred = y_pred * complement


            #let's calculate one crossentropy loss for each group
            #(directly from the keras loss functions imported above)
            falsePosLoss = binary_crossentropy(falsePosGroupTrue,falsePosGroupPred)
            nonFalseLoss = binary_crossentropy(nonFalseGroupTrue,nonFalseGroupPred)

            #return them weighted:
            return (false_positive_weight*falsePosLoss) + (nonFalseLoss)

        ncce = functools.partial(w_categorical_crossentropy, weights=self.train_dataset.class_weight)
        # ncce = functools.partial(weighted_binary_crossentropy)
        model_params = {
            'loss': 'categorical_crossentropy',
            # 'loss': weighted_binary_crossentropy,
            'optimizer': Adam(beta_1=0.9,
                         beta_2=0.99,
                         epsilon=1e-08,
                         decay=0.0,
                         amsgrad=True,
                         # clipnorm=self.gradclip_value
                         ),
            'metrics': [categorical_accuracy]
        }
        self.modelconfig = self.model.compile(**model_params)
        print("Available metrics: ", self.model.metrics_names)
        print("false_positive_weight = ", self.train_dataset.class_weight[0])        
        print("false_negative_weight = ", self.train_dataset.class_weight[1])

        tempfilepath = os.path.join(self.explogdir, "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5")

        '''                         CREATE CALLBACKS                        '''
        # callbacks availabble
        checkpoint = ModelCheckpoint(tempfilepath,
                                     # monitor='val_acc',
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
                                    embeddings_freq=0, 
                                    embeddings_layer_names=None, 
                                    embeddings_metadata=None, 
                                    embeddings_data=None)
        metrichistory = MetricsCallback()
        self.callbacks = [
                        # checkpoint,
                        # reduce_lr,
                        # tboard,
                        metrichistory
                    ]

    def test(self, modelname):
        def predict_with_uncertainty(f, x, n_iter=10):
            result = np.zeros((n_iter,) + x.shape)

            for iter in range(n_iter):
                result[iter] = f(x, 1)

            prediction = result.mean(axis=0)
            uncertainty = result.var(axis=0)
            return prediction, uncertainty

        # create a "learning phase" function, to allow prediction with uncertainty
        f = K.Function(self.model.inputs + [K.learning_phase()], self.model.outputs)
        X = [self.test_dataset.X_aux, self.test_dataset.X_chan]
        prediction, uncertainty = predict_with_uncertainty(f, X, n_iter=50)

        # determine scoring of these predictions
        metricsfilepath = os.path.join(self.outputdatadir, modelname+ "_test_predictions.json")

        metricdata = {
            'prediction': prediction,
            'uncertainty': uncertainty
        }
        self._writejsonfile(metricdata, metricsfilepath)

    def train(self):
        self._loadgenerator()
        print("Training data: ", self.train_dataset.X_aux.shape,  self.train_dataset.ylabels.shape)
        print("Testing data: ",  self.test_dataset.X_aux.shape,  self.test_dataset.ylabels.shape)
        print("Class weights are: ",  self.train_dataset.class_weight)
        test = np.argmax( self.train_dataset.ylabels, axis=1)
        print("class imbalance: ", np.sum(test), len(test))

        # swap the class weights
        # self.train_dataset.class_weight[[0,1]] = self.train_dataset.class_weight[[1,0]]

        self.AUGMENT = False
        # self.steps_per_epoch = 2
        # augment data, or not and then trian the model!
        if not self.AUGMENT:
            print('Not using data augmentation. Implement Solution still!')
            HH = self.model.fit(self.train_dataset.X_chan,
                # [self.train_dataset.X_aux, self.train_dataset.X_chan], 
                              self.train_dataset.ylabels,
                              batch_size = self.batch_size,
                              epochs=self.num_epochs,
                              # validation_data=([self.test_dataset.X_aux, self.test_dataset.X_chan], self.test_dataset.ylabels),
                              validation_data=(self.test_dataset.X_chan, self.test_dataset.ylabels),
                              shuffle=self.shuffle,
                              # class_weight= self.train_dataset.class_weight,
                              callbacks=self.callbacks, verbose=2)
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
                                        validation_data=([self.test_dataset.X_aux, self.test_dataset.X_chan], self.test_dataset.ylabels),
                                        shuffle=self.shuffle,
                                        class_weight= self.train_dataset.class_weight,
                                        callbacks=self.callbacks, verbose=2)
        else:
            print('Using real-time data augmentation.')
            # self.generator.fit(X_train)
            HH = self.model.fit_generator(self.generator.flow([self.train_dataset.X_aux, 
                                                            self.train_dataset.X_chan], 
                                                            self.train_dataset.ylabels, 
                                                            batch_size=self.batch_size),
                                        steps_per_epoch=self.steps_per_epoch,
                                        epochs=self.num_epochs,
                                        validation_data=([self.test_dataset.X_aux, self.test_dataset.X_chan], self.test_dataset.ylabels),
                                        shuffle=self.shuffle,
                                        class_weight= self.train_dataset.class_weight,
                                        callbacks=self.callbacks, verbose=2)

        self.HH = HH
        self.metrichistory = self.callbacks[0] 

    def _loadgenerator(self):
        imagedatagen_args = {
            'featurewise_center':False,  # set input mean to 0 over the dataset
            'samplewise_center':False,  # set each sample mean to 0
            'featurewise_std_normalization':False,  # divide inputs by std of the dataset
            'samplewise_std_normalization':False,  # divide each input by its std
            'zca_whitening':False,      # apply ZCA whitening
            'rotation_range':2,         # randomly rotate images in the range (degrees, 0 to 180)
            'width_shift_range':0.2,    # randomly shift images horizontally (fraction of total width)
            'height_shift_range':0.2,   # randomly shift images vertically (fraction of total height)
            'horizontal_flip':True,    # randomly flip images
            'vertical_flip':True,      # randomly flip images
            'channel_shift_range':0,
            'fill_mode':'nearest',
            'preprocessing_function':Augmentations.preprocess_imgwithnoise
        }

        # This will do preprocessing and realtime data augmentation:
        self.generator = AuxImgDataGenerator(**imagedatagen_args)

    def _loadgeneratordir(self):
        pass