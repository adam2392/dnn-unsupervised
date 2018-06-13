import os
import numpy as np
import dnn
from datetime import datetime

class GenericConfig(object):
    _module_path = os.path.dirname(dnn.__file__)

def walkdir(directory):
    filepaths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.startswith('.'):
                filepaths.append(os.path.join(root, file))
    return filepaths

class InputConfig(object):
    '''
    Define the input configuration for the location of the raw
    data files to be used in training
    '''
    _base_input = os.getcwd()
    _train_data_files = []
    _test_data_files = []

    @property
    def RAW_DATA_FOLDER(self):
        if self._raw_data is not None:
            return self._raw_data

        # Expecting to run in the top of stats GIT repo, with the dummy head
        return os.path.join(self._base_input, "data", "raw")

    @property
    def TRAIN_DATA_FOLER(self):
        if self._train_data_files:
            return self._train_data_files 

        print("Did not initialize training data folder! You should...")

    @property
    def TEST_DATA_FOLER(self):
        if self._test_data_files:
            return self._test_data_files 

        print("Did not initialize test data folder! You should...")
    
    def __init__(self, raw_folder=None, train_data_folder=None, test_data_folder=None):
        self._raw_data = raw_folder
        self._train_data = train_data_folder
        self._test_data = test_data_folder
        
        if train_data_folder is not None:
            self._train_data_files = walkdir(train_data_folder)
        if test_data_folder is not None:
            self._train_data_files = walkdir(test_data_folder)

class TensorboardConfig(object):
    subfolder = None

    def __init__(self, out_base=None, separate_by_run=False):
        """
        :param work_folder: Base folder where logs/figures/results should be kept
        :param separate_by_run: Set TRUE, when you want logs/results/figures to be in different files / each run
        """
        self._out_base = out_base or os.path.join(
            os.getcwd(), "tensorboard_out")
        self._separate_by_run = separate_by_run

    @property
    def FOLDER_LOGS(self):
        folder = os.path.join(self._out_base, "logs")
        if self._separate_by_run:
            folder = folder + \
                datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')
        if not (os.path.isdir(folder)):
            os.makedirs(folder)
        return folder

class OutputConfig(object):
    subfolder = None

    def __init__(self, out_base=None, separate_by_run=False):
        """
        :param work_folder: Base folder where logs/figures/results should be kept
        :param separate_by_run: Set TRUE, when you want logs/results/figures to be in different files / each run
        """
        self._out_base = out_base or os.path.join(
            os.getcwd(), "_dnn_pytorch_out")
        self._separate_by_run = separate_by_run

    @property
    def FOLDER_LOGS(self):
        folder = os.path.join(self._out_base, "logs")
        if self._separate_by_run:
            folder = folder + \
                datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')
        if not (os.path.isdir(folder)):
            os.makedirs(folder)
        return folder

    @property
    def FOLDER_RES(self):
        folder = os.path.join(self._out_base, "res")
        if self._separate_by_run:
            folder = folder + \
                datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')
        if not (os.path.isdir(folder)):
            os.makedirs(folder)
        if self.subfolder is not None:
            os.path.join(folder, self.subfolder)
        return folder

    @property
    def FOLDER_FIGURES(self):
        folder = os.path.join(self._out_base, "figs")
        if self._separate_by_run:
            folder = folder + \
                datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')
        if not (os.path.isdir(folder)):
            os.makedirs(folder)
        if self.subfolder is not None:
            os.path.join(folder, self.subfolder)
        return folder

    @property
    def FOLDER_TEMP(self):
        return os.path.join(self._out_base, "temp")


class CalculusConfig(object):
    SYMBOLIC_CALCULATIONS_FLAG = False
    # Normalization configuration
    WEIGHTS_NORM_PERCENT = 95
    MIN_SINGLE_VALUE = np.finfo("single").min
    MAX_SINGLE_VALUE = np.finfo("single").max
    MAX_INT_VALUE = np.iinfo(np.int64).max
    MIN_INT_VALUE = np.iinfo(np.int64).max


class FiguresConfig(object):
    VERY_LARGE_SIZE = (40, 20)
    VERY_LARGE_PORTRAIT = (30, 50)
    SUPER_LARGE_SIZE = (80, 40)
    LARGE_SIZE = (20, 15)
    SMALL_SIZE = (15, 10)
    FIG_FORMAT = 'png'  # 'eps' 'pdf' 'svg'
    SAVE_FLAG = True
    SHOW_FLAG = True
    MOUSE_HOOVER = False
    MATPLOTLIB_BACKEND = "Qt4Agg"  # , "Agg", "qt5"


class Config(object):
    generic = GenericConfig()
    figures = FiguresConfig()
    calcul = CalculusConfig()

    def __init__(self,
                 raw_data_folder=None,
                 output_base=None,
                 separate_by_run=False):
        self.input = InputConfig(raw_data_folder)
        self.out = OutputConfig(output_base, separate_by_run)
        self.tboard = TensorboardConfig(output_base, separate_by_run)
