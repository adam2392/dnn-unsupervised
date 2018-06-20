import numpy as np
import pandas as pd
import json
import os

from dnn.base.constants.config import Config
from dnn.base.utils.log_error import initialize_logger
import dnn.base.constants.model_constants as constants
from dnn.io.dataloaders.base import BaseLoader
from sklearn.utils import compute_class_weight

# preprocessing data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

import tqdm 

class ReaderSeqDataset(BaseLoader):
    root_dir = None
    patients = None
    testfilepaths = None
    trainfilepaths = None
    filelist = None

    def __init__(self, config=None):
        super(ReaderSeqDataset, self).__init__(config=config)

    def __len__(self):
        return len(self.filelist)

    def load_filepaths(self, traindir, testdir, procedure='loo', testname=None):
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

    def loadfiles_list(self, seqlength, filelist=[], mode=constants.TRAIN):
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

        # perform the mining of data thru file list and append to a list 
        dataset_list = []
        labels_list = []
        labels_tmp = []
        # filerange = enumerate(filelist)
        filerange = enumerate(tqdm.tqdm(filelist))

        for idx, datafile in filerange:
            if not datafile.endswith('.npz'):
                datafile += '.npz'
            imagedata = np.load(datafile)
            _image_tensor = imagedata['image_tensor']
            _ylabels = imagedata['ylabels']

            _image_tensor = self._formatdata(_image_tensor)

            labels_tmp.extend(_ylabels.astype(int).ravel())
            dataset_list.append(_image_tensor)
            labels_list.append(_ylabels.astype(int).ravel())

        # form sequences
        seqdata, seqylabels = self.form_seq_data(dataset_list, labels_list, seqlength=seqlength)

        print(_ylabels.shape)
        print(_image_tensor.shape)
        print(dataset_list[0].shape)
        print(seqdata.shape)
        print(seqylabels.shape)
        
        # compute the class weights
        class_weight = compute_class_weight('balanced',
                            np.unique(labels_tmp).astype(int), labels_tmp)
        if mode == constants.TRAIN:
            self.train_dataset.X_train = seqdata
            self.train_dataset.y_train = seqylabels
            self.train_dataset.class_weight = class_weight

            self.logger.info("Finished setting training data!")
        elif mode == constants.TEST:
            self.test_dataset.X_test = seqdata
            self.test_dataset.y_test = seqylabels
            self.test_dataset.class_weight = class_weight

            self.logger.info("Finished setting testing data!")
    def form_seq_data(self, dataset_list, labels_list, seqlength):
        assert len(dataset_list) == len(labels_list)

        # initialize return sequence datasets
        seqdata = []
        seqylabels = []

        # loop through our list of unformatted data
        datarange = enumerate(tqdm.tqdm(dataset_list))
        for idx, dataset in datarange:
            labels = labels_list[idx]
            assert len(dataset) == len(labels)

            # unpack and format datasets
            formatted_dataset = list(map(np.array, zip(*(dataset[i:,...] for i in range(seqlength)))))
            formatted_labels = list(map(list, zip(*(labels[i:] for i in range(seqlength)))))

            # unpack the formatted datasets
            for i in range(len(formatted_dataset)):
                seqx = formatted_dataset[i]
                seqy = formatted_labels[i]
                seqdata.append(seqx)
                seqylabels.append(seqy)

        return seqdata, seqylabels

    def loadfiles_arr(self, seqlength, filelist=[], mode=constants.TRAIN):
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

        filerange = enumerate(filelist)
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
