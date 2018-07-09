import numpy as np
import pandas as pd
import json
import os

import dnn.base.constants.model_constants as constants
from dnn.io.dataloaders.baseaux import BaseAuxLoader
from dnn.io.dataloaders.basedataset import TrainDataset, TestDataset

# preprocessing data
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

class ReaderEZNetDataset(BaseAuxLoader):
    root_dir = None
    patients = None
    testfilepaths = None
    trainfilepaths = None
    filelist = None

    def __init__(self, config=None):
        super(ReaderEZNetDataset, self).__init__(config=config)

    def loadbydir(self, traindir, testdir):
        self.logger.info("Reading testing data directory %s " % testdir)
        self.logger.info("Reading training data directory %s " % traindir)

        ''' Get list of file paths '''
        self.testfilepaths = []
        for root, dirs, files in os.walk(testdir):
            for file in files:
                # file = file.split('.')[0]
                if file.endswith('.npz') \
                    and not file.startswith('.'):
                 # \
                    # and testname not in file:
                    self.testfilepaths.append(os.path.join(root, file))

        
        ''' Get list of file paths '''
        self.trainfilepaths = []
        for root, dirs, files in os.walk(traindir):
            for file in files:
                # file = file.split('.')[0]
                if file.endswith('.npz') \
                    and not file.startswith('.'):
                # \
                    # and testname not in file:
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

        def update_tensor(tensor):
            newtensor = np.zeros((tensor.shape[0] * 2, *tuple(tensor.shape[1:])))
            newtensor[0:tensor.shape[0], ...] = tensor
            return newtensor
            
        def pad_tensor(shape, tensor):
            result = np.zeros(shape)
            if tensor.ndim == 2:
                result[:,:tensor.shape[1]] = tensor
            elif tensor.ndim == 3:
                result[:,:tensor.shape[1],:tensor.shape[2]] = tensor
            return result

        filerange = enumerate(filelist)

        for idx, datafile in filerange:
            if not datafile.endswith('.npz'):
                datafile += '.npz'

            datastruct = np.load(datafile)
            _aux_tensor = datastruct['auxmats']
            _chan_tensor = datastruct['chanvectors']
            _ylabels = datastruct['ylabels']
            _chanlabel = datastruct['chanlabels']

            if _aux_tensor.shape[-1] > 480:
                _aux_tensor = _aux_tensor[...,0:480]
                _chan_tensor = _chan_tensor[...,0:480]
                _ylabels = _ylabels[...,0:480]

            # apply padding
            auxshape = (len(_aux_tensor),30,480)
            chanshape = (len(_chan_tensor),480)
            _aux_tensor = pad_tensor(auxshape, _aux_tensor)
            _chan_tensor = pad_tensor(chanshape, _chan_tensor)

            if idx == 0:
                aux_tensors = np.zeros(
                    (len(filelist) * 1000, *tuple(_aux_tensor.shape[1:])))
                chan_tensors = np.zeros(
                    (len(filelist) * 1000, *tuple(_chan_tensor.shape[1:])))
                ylabels = np.zeros(
                    (len(filelist) * 1000, *tuple(_ylabels.shape[1:])))
                wins = [0, 0 + _ylabels.shape[0]]
                prevwin = wins[-1]

            else:
                wins = [prevwin, prevwin + _ylabels.shape[0]]
                prevwin = wins[-1]

                if prevwin > aux_tensors.shape[0]:
                    aux_tensors = update_tensor(aux_tensors)
                    chan_tensors = update_tensor(chan_tensors)
                    ylabels = update_tensor(ylabels)

            aux_tensors[wins[0]:wins[1], ...] = _aux_tensor
            chan_tensors[wins[0]:wins[1], ...] = _chan_tensor
            ylabels[wins[0]:wins[1], ...] = _ylabels
            # break

        print(aux_tensors.shape)
        print(ylabels.shape)
        print(chan_tensors.shape)
        deleterange = np.arange(prevwin, len(aux_tensors))
        aux_tensors = np.delete(aux_tensors, deleterange, axis=0)
        ylabels = np.delete(ylabels, deleterange, axis=0)
        chan_tensors = np.delete(chan_tensors, deleterange, axis=0)

        # load the ylabeled data 1 in 0th position is 0, 1 in 1st position is 1
        if ylabels.ndim == 1:
            ylabels = ylabels[:,np.newaxis]
        if aux_tensors.ndim == 3:
            aux_tensors = aux_tensors[...,np.newaxis]
        if chan_tensors.ndim == 2:
            chan_tensors = chan_tensors[...,np.newaxis]

        # aux_tensors = self._formatdata(aux_tensors)
        invert_y = 1 - ylabels
        ylabels = np.concatenate((invert_y, ylabels), axis=1)
        # format the data correctly
        class_weight = compute_class_weight('balanced',
                                            np.unique(
                                                ylabels).astype(int),
                                            np.argmax(ylabels, axis=1))

        self.logger.info("Image tensor shape: {}".format(aux_tensors.shape))

        if mode == constants.TRAIN:
            self.train_dataset = TrainDataset(chan_tensors, ylabels, aux_tensors)
            # self.train_dataset.X_aux = aux_tensors
            # self.train_dataset.X = chan_tensors
            # self.train_dataset.y = ylabels
            self.train_dataset.class_weight = class_weight
        elif mode == constants.TEST:
            self.test_dataset = TestDataset(chan_tensors, ylabels, aux_tensors)
            # self.test_dataset.X_aux = aux_tensors
            # self.test_dataset.X = chan_tensors
            # self.test_dataset.y = ylabels
            self.test_dataset.class_weight = class_weight

