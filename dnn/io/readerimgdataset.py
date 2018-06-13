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

class TrainDataset(object):
    X_train = None
    y_train = None
    class_weight = None

    def __len__(self):
        return len(self.X_train)

    @property
    def imsize(self):
        return self.X_train.shape[2]

    @property
    def n_colors(self):
        return self.X_train.shape[3]

class TestDataset(object):
    X_test = None
    y_test = None
    class_weight = None

    def __len__(self):
        return len(self.X_test)

    @property
    def imsize(self):
        return self.X_train.shape[2]

    @property
    def n_colors(self):
        return self.X_train.shape[3]

class ReaderImgDataset(object):
    root_dir = None
    patients = None
    testfilepaths = None
    trainfilepaths = None
    filelist = None

    train_dataset = TrainDataset()
    test_dataset = TestDataset()
    # X_train = None
    # X_test = None
    # y_train = None
    # y_test = None
    train_class_weight = None
    test_class_weight = None

    def __init__(self, config=None):
        self.config = config or Config()
        self.logger = initialize_logger(
            self.__class__.__name__,
            self.config.out.FOLDER_LOGS)

    def __len__(self):
        return len(self.filelist)

    def loadbydir(self, traindir, testdir, procedure='loo', testname=None):
        self.logger.info("Reading testing data directory %s " % testdir)

        if procedure == 'loo' and testname is None:
            self.logger.error(
                "Testname must be set to the file directory we want to ignore!")
            return
        elif procedure == 'loo':
            self.logger.info("Getting test files for {}".format(testname))
            ''' Get list of file paths '''
            self.testfilepaths = []
            for root, dirs, files in os.walk(testdir):
                for file in files:
                    # file = file.split('.')[0]
                    if testname in file and file.endswith('.npz'):
                        self.testfilepaths.append(os.path.join(root, file))

            self.logger.info("Reading training data directory %s " % traindir)
            ''' Get list of file paths '''
            self.trainfilepaths = []
            for root, dirs, files in os.walk(traindir):
                for file in files:
                    # file = file.split('.')[0]
                    if testname not in file and file.endswith('.npz'):
                        self.trainfilepaths.append(os.path.join(root, file))
        else:
            ''' Get list of file paths '''
            self.testfilepaths = []
            for root, dirs, files in os.walk(testdir):
                for file in files:
                    file = file.split('.')[0]
                    self.testfilepaths.append(os.path.join(root, file))
            self.testfilepaths.append(os.path.join(root, file))

            self.logger.info("Reading training data directory %s " % traindir)
            ''' Get list of file paths '''
            self.trainfilepaths = []
            for root, dirs, files in os.walk(traindir):
                for file in files:
                    file = file.split('.')[0]
                    self.trainfilepaths.append(os.path.join(root, file))

        self.logger.info("Found {} training files and {} testing files.".format(len(self.trainfilepaths), len(self.testfilepaths)))
        self.logger.info("Finished reading in data by directories!")

    def loadfiles(self, filelist=[], mode=constants.TRAIN):
        if len(filelist) == 0:
            if self.trainfilepaths is not None and mode == constants.TRAIN:
                filelist = self.trainfilepaths
            elif self.testfilepaths is not None and mode == constants.TEST:
                filelist = self.testfilepaths
            else:
                self.logger.info(
                    "Mode: {} and filelist: {}".format(
                        mode, filelist))
                self.logger.error(
                    "Need to either load filepaths, or pass in correct mode!")
            self.logger.info("Loading files from directory!")
        else:
            self.logger.info("Loading files from user passed in files!")

        # print(self.trainfilepaths)
        # print(self.testfilepaths)
        '''     LOAD DATA      '''
        # try:
        # filerange = enumerate(tqdm.trange(filelist))
        # except Exception as e:
        filerange = enumerate(filelist)
        #     print(e)

        for idx, datafile in filerange:
            if not datafile.endswith('.npz'):
                datafile += '.npz'
            imagedata = np.load(datafile)
            _image_tensor = imagedata['image_tensor']
            _ylabels = imagedata['ylabels']

            if idx == 0:
                image_tensors = np.zeros(
                    (len(filelist) * 1000, *tuple(_image_tensor.shape[1:])))
                ylabels = np.zeros(
                    (len(filelist) * 1000, *tuple(_ylabels.shape[1:])))
                wins = [0, 0 + _ylabels.shape[0]]
                prevwin = wins[-1]

                image_tensors[wins[0]:wins[1], ...] = _image_tensor
                ylabels[wins[0]:wins[1], ...] = _ylabels
                # image_tensors = _image_tensor
                # ylabels = _ylabels
            else:
                wins = [prevwin, prevwin + _ylabels.shape[0]]
                prevwin = wins[-1]

                if prevwin > image_tensors.shape[0]:
                    new_image_tensors = np.zeros(
                        (image_tensors.shape[0] * 2, *tuple(image_tensors.shape[1:])))
                    new_ylabels = np.zeros(
                        (ylabels.shape[0] * 2, *tuple(ylabels.shape[1:])))
                    new_image_tensors[0:image_tensors.shape[0], ...] = image_tensors
                    new_ylabels[0:ylabels.shape[0], ...] = ylabels
                    image_tensors = new_image_tensors
                    ylabels = new_ylabels

                image_tensors[wins[0]:wins[1], ...] = _image_tensor
                ylabels[wins[0]:wins[1], ...] = _ylabels
                # image_tensors = np.append(image_tensors, _image_tensor, axis=0)
                # ylabels = np.append(ylabels, _ylabels, axis=0)

            # break

        deleterange = np.arange(prevwin, len(image_tensors))
        image_tensors = np.delete(image_tensors, deleterange, axis=0)
        ylabels = np.delete(ylabels, deleterange, axis=0)

        # load the ylabeled data 1 in 0th position is 0, 1 in 1st position is 1
        invert_y = 1 - ylabels
        ylabels = np.concatenate((invert_y, ylabels), axis=1)
        # format the data correctly
        class_weight = compute_class_weight('balanced',
                                            np.unique(
                                                ylabels).astype(int),
                                            np.argmax(ylabels, axis=1))

        image_tensors = self._formatdata(image_tensors)
        self.logger.info("Image tensor shape: {}".format(image_tensors.shape))

        if mode == constants.TRAIN:
            self.train_dataset.X_train = image_tensors
            self.train_dataset.y_train = ylabels
            self.train_dataset.class_weight = class_weight
        elif mode == constants.TEST:
            self.test_dataset.X_test = image_tensors
            self.test_dataset.y_test = ylabels
            self.test_dataset.class_weight = class_weight

    def _formatdata(self, images):
        images = images.swapaxes(1, 3)
        # lower sample by casting to 32 bits
        images = images.astype("float32")
        return images

    @property
    def imsize(self):
        return self.train_dataset.X_train.shape[2]

    @property
    def n_colors(self):
        return self.train_dataset.X_train.shape[3]

    def getchanstats(self):
        chanaxis = 3
        numchans = self.train_dataset.X_train.shape[chanaxis]

        chanmeans = []
        chanstd = []
        for ichan in range(numchans):
            chandata = self.train_dataset.X_train[...,ichan].ravel()
            chanmeans.append(np.mean(chandata))
            chanstd.append(np.std(chandata))
        self.chanmeans = np.array(chanmeans)
        self.chanstd = np.array(chanstd)