# coding=utf-8

import numpy as np
from enum import Enum
from fragility.base.utils.data_structures_utils import reg_dict, formal_repr, \
    sort_dict, labels_to_inds
from fragility.base.computations.math_utils import normalize_weights
from fragility.base.model.fragilitymodel import FragilityModel
import fragility.base.constants.model_constants as constants


class HeatmapTypes(Enum):
    TYPE_FRAGILITY = 'FRAGILITY'
    TYPE_POWERSPECT = "POWERSPECTRUM"


class HeatmapH5Field(object):
    FRAGILITY = "fragility"
    LABELS = "region_labels"


class Heatmap(object):
    TYPE_FRAGILITY = HeatmapTypes.TYPE_FRAGILITY.value
    TYPE_POWERSPECT = HeatmapTypes.TYPE_POWERSPECT.value

    file_path = None
    matrix_over_time = None
    fragility = None
    minnorm_perturbation = None
    labels = None
    locations = None
    timepoints = None
    h_type = TYPE_FRAGILITY

    def __init__(self, file_path, matrix_over_time,
                 labels=np.array([]), locations=np.array([]),
                 timepoints=np.array([]), freqband=constants.GAMMA):
        self.file_path = file_path
        self.labels = labels
        self.locations = locations
        self.timepoints = timepoints
        self.h_type = h_type

        # self.freqband = freqband
        # if h_type is TYPE_POWERSPECT:
        #     if not freqband:
        #         print("need to set freqband!")
        #     self.matrix_over_time = matrix_over_time
        #     self.freqband = freqband
        # elif h_type is TYPE_FRAGILITY:
        self.minnorm_perturbation = matrix_over_time
        self.fragility = FragilityModel.compute_fragilitymetric(
            minnorm_perturbation)

    @property
    def number_of_regions(self):
        return len(self.labels)

    def __repr__(self):
        d = {"f. minimum norm perturbation": reg_dict(self.minnorm_perturbation, self.labels),
             "g. fragility": reg_dict(self.fragility, self.labels),
             "a. labels": reg_dict(self.labels),
             "b. locations": reg_dict(self.locations, self.labels)}
        return formal_repr(self, sort_dict(d))

    def __str__(self):
        return self.__repr__()

    def regions_labels2inds(self, labels):
        inds = []
        for lbl in labels:
            inds.append(np.where(self.labels == lbl)[0][0])
        if len(inds) == 1:
            return inds[0]
        else:
            return inds

    def get_regions_inds_by_labels(self, labels):
        return labels_to_inds(self.labels, labels)

    @staticmethod
    def threshmap(mat, thresh):
        mat = mat.copy()
        mat[mat < thresh] = 0
        return mat
