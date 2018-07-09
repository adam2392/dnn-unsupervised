import numpy as np
import pandas as pd
import json
import os

from dnn.base.constants.config import Config
from dnn.base.utils.log_error import initialize_logger
import dnn.base.constants.model_constants as constants
from dnn.io.dataloaders.base import BaseLoader
from sklearn.utils import compute_class_weight

from dnn.io.dataloaders.basedataset import TrainDataset, TestDataset

# preprocessing data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

ProcedureTypes = ['loo', 'naive']

class ReaderImgDataset(BaseLoader):
    root_dir = None
    patients = None
    testfilepaths = None
    trainfilepaths = None
    filelist = None

    def __init__(self, config=None):
        super(ReaderImgDataset, self).__init__(config=config)

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
        elif procedure == 'naive':
            from sklearn.model_selection import train_test_split
            
            # test and train directory are separate
            if testdir != traindir:
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
            # test and train directory are the same, so do a train/test split
            else:
                allfilepaths = []
                for root, dirs, files in os.walk(testdir):
                    for file in files:
                        file = file.split('.')[0]
                        allfilepaths.append(os.path.join(root, file))
                self.trainfilepaths, self.testfilepaths = train_test_split(allfilepaths, test_size=0.33, random_state=42)

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
            # dirname = os.path.dirname(datafile)
            # jsonfilename = dirname.split('__stftimgmodel')[0] + '_meta.json'
            
            imagedata = np.load(datafile)
            _image_tensor = imagedata['image_tensor']
            _ylabels = imagedata['ylabels']

            # print(datafile)
            # print(_image_tensor.shape)
            # print(_ylabels.shape)
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
            self.train_dataset = TrainDataset(image_tensors, ylabels)
            # self.train_dataset.X_aux = aux_tensors
            # self.train_dataset.X = chan_tensors
            # self.train_dataset.y = ylabels
            self.train_dataset.class_weight = class_weight
        elif mode == constants.TEST:
            self.test_dataset = TestDataset(image_tensors, ylabels)
            # self.test_dataset.X_aux = aux_tensors
            # self.test_dataset.X = chan_tensors
            # self.test_dataset.y = ylabels
            self.test_dataset.class_weight = class_weight

