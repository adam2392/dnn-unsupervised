import numpy as np
from dnn.base.constants.config import Config, OutputConfig
from dnn.base.utils.log_error import initialize_logger

class Postalarm(object):
    def __init__(self, winsize, stepsize, config=None):
        self.config = config or Config()
        self.logger = initialize_logger(
            self.__class__.__name__,
            self.config.out.FOLDER_LOGS)

        self.winsize = winsize
        self.stepsize = stepsize

    def load_predictions(self, seqoutputs):
        self.seqoutputs = seqoutputs
        numsamples = len(seqoutputs)
        self.compute_samplepoints(numsamples)

    def temporal_smooth(self, thresh=0.5):
        """
        Function to temporally smooth classification outputs
        using a firepower metric that is simply thresholded at 0.5.

        Returns alarms generated over N milliseconds
        """
        # initialize firing power of classifier output
        fpower = []
        alarm_inds = []
        ALARMSET = False

        # go through windows
        for idx, win in enumerate(self.samplepoints):
            lowend = win[0]
            highend = win[1]

            # sum all predicted outputs / length of this range
            curr_fpower = np.sum(
                self.seqoutputs[lowend:highend]) / self.winsamps

            # once we've passed a window, we can raise another alarm if we find
            # it
            if alarm_inds:
                if idx > alarm_inds[-1] + self.winsamps:
                    ALARMSET = False

            # check if we should set alarm_inds
            if curr_fpower >= thresh and not ALARMSET:
                alarm_inds.append(idx)
                ALARMSET = True

            # keep track of the firing powers
            fpower.append(curr_fpower)

        # find indices where threshold is exceeded
        # alarm_inds = np.where(fpower >= thresh)[0]
        return alarm_inds, self.timepoints.ravel()[-1]

    def compute_timepoints(self, winsize_persamp,
                           stepsize_persamp, numtimepoints, copy=True):
        # Creates a [n,2] array that holds the time range of each window
        # in the analysis over sliding windows.
        # trim signal and then convert into milliseconds
        # create array of indices of window start and end times
        timestarts = np.arange(
            0,
            numtimepoints -
            winsize_persamp +
            1,
            stepsize_persamp)
        timeends = np.arange(
            winsize_persamp - 1,
            numtimepoints,
            stepsize_persamp)
        # create the timepoints array for entire data array
        timepoints = np.append(timestarts[:, np.newaxis],
                               timeends[:, np.newaxis], axis=1)
        if copy:
            self.timepoints = timepoints
        else:
            return timepoints

    def compute_samplepoints(self, numtimepoints, copy=True):
        # Creates a [n,2] array that holds the sample range of each window that
        # is used to index the raw data for a sliding window analysis
        samplestarts = np.arange(
            0,
            numtimepoints -
            self.winsize +
            1.,
            self.stepsize).astype(int)
        sampleends = np.arange(
            self.winsize - 1.,
            numtimepoints,
            self.stepsize).astype(int)
        samplepoints = np.append(samplestarts[:, np.newaxis],
                                 sampleends[:, np.newaxis], axis=1)
        self.numwins = samplepoints.shape[0]
        if copy:
            self.samplepoints = samplepoints
        else:
            return samplepoints
