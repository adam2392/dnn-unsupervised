# coding=utf-8

import numpy as np
import json
import pandas as pd
from enum import Enum
from dnn_pytorch.base.utils.data_structures_utils import reg_dict, formal_repr, \
    sort_dict, labels_to_inds
import dnn_pytorch.base.constants.model_constants as constants

from dnn_pytorch.base.constants.config import Config
from dnn_pytorch.base.utils.log_error import initialize_logger

# np.random.seed(123)
from sklearn.preprocessing import scale


class ImageseqTypes(Enum):
    TYPE_FRAGILITY = 'FRAGILITY'
    TYPE_POWERSPECT = "POWERSPECTRUM"


class Imageseq(object):
    TYPE_FRAGILITY = ImageseqTypes.TYPE_FRAGILITY
    TYPE_POWERSPECT = ImageseqTypes.TYPE_POWERSPECT
    h_type = TYPE_POWERSPECT

    file_path = None
    datafile = None
    labels = None
    locations = None
    timepoints = None
    patient = None
    imsize = None
    ylabels = None
    image_tensor = None

    def __init__(self, file_path, datafile, patient=None, config=None):
        self.config = config or Config()
        self.logger = initialize_logger(
            self.__class__.__name__,
            self.config.out.FOLDER_LOGS)

        self.file_path = file_path
        self.datafile = datafile
        self.patient = patient

        self.metafile = datafile.split('.npz')[0] + '.json'

        self._loaddata()

    def __repr__(self):
        d = {"f. ylabels": self.ylabels,
             "g. image_tensor": reg_dict(self.image_tensor.shape, self.ylabels),
             "b. imsize": self.imsize}
        return formal_repr(self, sort_dict(d))

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def threshmap(mat, thresh):
        mat = mat.copy()
        mat[mat < thresh] = 0
        return mat

    def _loaddata(self):
        self.logger.info("Reading data struct from %s" % self.datafile)

        datastruct = np.load(self.datafile)
        self.image_tensor = datastruct['image_tensor']
        self.ylabels = datastruct['ylabels']

        # load in metadata
        with open(self.metafile, encoding='utf-8') as fp:
            jsonstr = json.loads(fp.read())
        self.metadata = json.loads(jsonstr)

        self.logger.info("Finished reading in data successfully!")
