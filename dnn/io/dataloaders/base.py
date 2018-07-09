import numpy as np
import pandas as pd
import json
import os

from dnn.base.constants.config import Config
from dnn.base.utils.log_error import initialize_logger
import dnn.base.constants.model_constants as constants
from sklearn.utils import compute_class_weight

# preprocessing data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

class BaseLoader(object):
    root_dir = None
    patients = None
    testfilepaths = None
    trainfilepaths = None
    filelist = None

    train_dataset = None
    test_dataset = None
    # train_dataset = TrainDataset()
    # test_dataset = TestDataset()

    def __init__(self, config=None):
        self.config = config or Config()
        self.logger = initialize_logger(
            self.__class__.__name__,
            self.config.out.FOLDER_LOGS)

    def __len__(self):
        return len(self.filelist)

    def loadbydir(self, traindir, testdir, procedure='loo', testname=None):
        raise NotImplementedError("Need to implement function to load filepaths\
                for all datasets into list.")

    def loadfiles(self, filelist=[], mode=constants.TRAIN):
        raise NotImplementedError("Need to implement function to load files into memory.")

    def _formatdata(self, images):
        images = images.swapaxes(1, 3)
        # lower sample by casting to 32 bits
        images = images.astype("float32")
        return images

    def getchanstats(self, chanaxis=3):
        '''
        Chan axis = 3 if using keras/tensorflow
        Chan axis = 1 if using pytorch
        '''
        numchans = self.train_dataset.X_train.shape[chanaxis]

        chanmeans = []
        chanstd = []
        for ichan in range(numchans):
            chandata = self.train_dataset.X_train[...,ichan].ravel()
            chanmeans.append(np.mean(chandata))
            chanstd.append(np.std(chandata))
        self.chanmeans = np.array(chanmeans)
        self.chanstd = np.array(chanstd)
