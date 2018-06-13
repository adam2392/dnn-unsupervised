# Imports necessary for this function
from dnn_pytorch.base.constants.config import Config
from dnn_pytorch.base.utils.log_error import initialize_logger
from dnn_pytorch.base.utils.data_structures_utils import reg_dict, formal_repr, \
    sort_dict, NumpyEncoder
import dnn_pytorch.base.constants.model_constants as constants


class BaseMetric(object):
    accuracy = None     # chance of getting a correct answer
    # defined as the number of true positives (T_p) over the number of true
    # positives plus the number of false positives (F_p).
    precision = None
    # Recall (R) is defined as the number of true positives (T_p) over the
    # number of true positives plus the number of false negatives (F_n).
    recall = None
    fpr_m = None        # the rate of false positives per minute
    fpr_h = None        # the rate of false positives per hour

    doa = None          # degree of agreement
    baseline_acc = None  # accuracy of the imbalanced class

    def __init__(self, config=None):
        self.config = config or Config()
        self.logger = initialize_logger(
            self.__class__.__name__,
            self.config.out.FOLDER_LOGS)

    def __repr__(self):
        d = {"accuracy": self.accuracy,
             "precision": self.precision,
             "recall": self.recall,
             "fpr_m": self.fpr_m,
             "fpr_h": self.fpr_h,
             "doa": self.doa}
        return formal_repr(self, sort_dict(d))

    def __str__(self):
        return self.__repr__()

    def ensure_1d(self, ylabs):
        return np.argmax(ylabs, axis=1)
