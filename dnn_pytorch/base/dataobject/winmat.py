# coding=utf-8

import numpy as np
from fragility.base.utils.data_structures_utils import reg_dict, formal_repr, sort_dict, labels_to_inds
from fragility.base.computations.math_utils import normalize_weights

class WinmatH5Field(object):
    STATE_TRANSITION_MATRIX = "win_matrix"
    LABELS = "labels"
    TIMEPOINTS = "timepoints"

class Winmat(object):
    file_path = None
    win_matrix = None
    labels = None
    timepoints = None

    def __init__(self, file_path, 
                win_matrix, 
                labels=np.array([]), 
                timepoints=np.array([])):
        self.file_path = file_path
        self.win_matrix = win_matrix
        self.labels = labels
        self.timepoints = timepoints

        assert len(timepoints) == win_matrix.shape[0]

        self.numwins = len(timepoints)

    @property
    def number_of_regions(self):
        return len(self.labels)

    def __repr__(self):
        d = {"g. win_matrix": reg_dict(self.win_matrix, self.timepoints),
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

    def get_regions_inds_by_labels(self, lbls):
        return labels_to_inds(self.labels, lbls)
