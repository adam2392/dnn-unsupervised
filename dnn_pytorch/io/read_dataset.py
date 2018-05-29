import numpy as np 
import pandas as pd 
import json 
import os
# from enum import Enum

from dnn_pytorch.base.constants.config import Config
from dnn_pytorch.base.utils.log_error import initialize_logger
import dnn_pytorch.base.constants.model_constants as constants

class Reader(object):
	root_dir = None
	patients = None
    testfilepaths = None
    trainfilepaths = None
    filelist = None 

	X_train = None
	X_test = None
	y_train = None
	y_test = None
	train_class_weight = None
	test_class_weight = None

	def __init__(self, config=None):
		self.config = config or Config()
        self.logger = initialize_logger(self.__class__.__name__, self.config.out.FOLDER_LOGS)

    def __len__(self):
        return len(self.filelist)

    def loadbydir(self, traindir, testdir):
    	self.logger.info("Reading testing data directory %s " % testdir)
    	''' Get list of file paths '''
        self.testfilepaths = []
        for root, dirs, files in os.walk(testdir):
            for file in files:
                self.testfilepaths.append(os.path.join(root, file))
        self.testfilepaths.append(os.path.join(root, file))

        self.logger.info("Reading training data directory %s " % traindir)
        ''' Get list of file paths '''
        self.trainfilepaths = []
        for root, dirs, files in os.walk(traindir):
            for file in files:
                self.trainfilepaths.append(os.path.join(root, file))

        self.logger.info("Finished reading in data by directories!")

    def loadfiles(self, filelist=[], mode=constants.TRAIN):
        if len(filelist) == 0:
            if self.trainfilepaths is not None and mode == constants.TRAIN:
                filelist = self.trainfilepaths
            elif self.testfilepaths is not None and mode == constants.TEST:
                filelist = self.testfilepaths
            else:
                self.logger.info("Mode: {} and filelist: {}".format(mode, filelist))
                self.logger.error("Need to either load filepaths, or pass in correct mode!")
            self.logger.info("Loading files from directory!")
        else:
            self.logger.info("Loading files from user passed in files!")

    	'''     LOAD DATA      '''
        for idx, datafile in enumerate(filelist):
            imagedata = np.load(datafile)
            _image_tensor = imagedata['image_tensor']
            _ylabels = imagedata['ylabels']

            if idx == 0:
                image_tensors = _image_tensor
                ylabels = _ylabels
            else:
                image_tensors = np.append(image_tensors, _image_tensor, axis=0)
                ylabels = np.append(ylabels, _ylabels, axis=0)


        # load the ylabeled data 1 in 0th position is 0, 1 in 1st position is 1
        invert_y = 1 - ylabels
        ylabels = np.concatenate((invert_y, ylabels), axis=1)
        # format the data correctly
        class_weight = sklearn.utils.compute_class_weight('balanced',
                                                          np.unique(
                                                              ylabels).astype(int),
                                                          np.argmax(ylabels, axis=1))
     	
        image_tensors = self._formatdata(image_tensors)
        self.logger.info("Image tensor shape: {}".format(image_tensors.shape))

        if mode == constants.TRAIN:
	        self.X_train = image_tensors
	        self.y_train = ylabels
	        self.train_class_weight = class_weight
	    elif mode == constants.TEST:
	    	self.X_test = image_tensors
        	self.y_test = ylabels
        	self.test_class_weight = class_weight

    def _formatdata(self, images):
        images = images.swapaxes(1, 3)
        # lower sample by casting to 32 bits
        images = images.astype("float32")
        return images


