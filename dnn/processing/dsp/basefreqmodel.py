'''
Base Models for the Fragility Module
# Authors: Adam Li
# Edited by: Adam Li
'''
# Imports necessary for this function
import numpy as np
import warnings

class BaseWindowModel(object):
    '''
    This is a base class wrapper for any type of window model that we use.
    For example, we use a sliding window to compute a MVAR-1 model, or to compute
    FFT models.

    @params:
    - model         (object)
    - winsize       (int) an int of the window size to use in milliseconds
    - stepsize      (int) an int of the window step size to use in milliseconds
    - samplerate    (float) the samplerate in Hz
    '''
    def __init__(self, winsizems=None, stepsizems=None, samplerate=None):
        # if not model:
        #     warnings.warn("Model was not set! Please initialize a model and pass it in as model=")
        if not winsizems:
            warnings.warn(
                "Window size was not set for sliding window model. Set a winsize in ms!")
        if not stepsizems:
            warnings.warn(
                "Step size was not set for sliding window model. Set a stepsize in ms!")
        if not samplerate:
            warnings.warn("User needs to pass in sample rate in Hz!")

        assert isinstance(winsizems, int)
        assert isinstance(stepsizems, int)

        # self.model = model
        self.winsize = winsizems
        self.stepsize = stepsizems
        self.samplerate = samplerate

        # compute the number of samples in window and step
        self._setsampsinwin()
        self._setsampsinstep()

    def _setsampsinwin(self):
        # onesamp_ms = 1. * 1000./self.samplerate
        # numsampsinwin = self.winsize / onesamp_ms
        self.winsamps = self.winsize * self.samplerate / 1000.
        if self.winsamps % 1 != 0:
            warnings.warn("The number of samples within your window size is not an even integer.\
                          Consider increasing/changing the window size.")

    def _setsampsinstep(self):
        self.stepsamps = self.stepsize * self.samplerate / 1000.
        if self.stepsamps % 1 != 0:
            warnings.warn("The number of samples within your step size is not an even integer.\
                          Consider increasing/changing the step size.")

    def compute_timepoints(self, numtimepoints, copy=True):
        # Creates a [n,2] array that holds the time range of each window
        # in the analysis over sliding windows.

        # trim signal and then convert into milliseconds
        # numtimepoints = numtimepoints - numtimepoints%(self.samplerate/6)
        timepoints_ms = numtimepoints * 1000. / self.samplerate

        # create array of indices of window start and end times
        timestarts = np.arange(
            0,
            timepoints_ms -
            self.winsize +
            1,
            self.stepsize)
        timeends = np.arange(self.winsize - 1, timepoints_ms, self.stepsize)
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
            self.winsamps +
            1.,
            self.stepsamps).astype(int)
        sampleends = np.arange(
            self.winsamps - 1.,
            numtimepoints,
            self.stepsamps).astype(int)
        samplepoints = np.append(samplestarts[:, np.newaxis],
                                 sampleends[:, np.newaxis], axis=1)
        self.numwins = samplepoints.shape[0]
        if copy:
            self.samplepoints = samplepoints
        else:
            return samplepoints


class BaseFreqModel(BaseWindowModel):
    def __init__(self, winsizems, stepsizems, samplerate):
        BaseWindowModel.__init__(self, winsizems=winsizems,
                                 stepsizems=stepsizems,
                                 samplerate=samplerate)

    def buffer(self, x, n, p=0, opt=None):
        '''Mimic MATLAB routine to generate buffer array

        MATLAB docs here: https://se.mathworks.com/help/signal/ref/buffer.html

        Args
        ----
        x:   signal array
        n:   number of data segments
        p:   number of values to overlap
        opt: initial condition options. default sets the first `p` values
             to zero, while 'nodelay' begins filling the buffer immediately.
        '''
        if p >= n:
            raise ValueError('p ({}) must be less than n ({}).'.format(p, n))
        assert isinstance(n, int)
        assert isinstance(p, int)

        # Calculate number of columns of buffer array
        if p == 0:
            cols = int(np.floor(len(x)/float(n-p)))
        else:
            cols = int(np.floor(len(x)/float(p)))

        # Check for opt parameters
        if opt == 'nodelay':
            # Need extra column to handle additional values left
            cols -= 1
        elif opt != None:
            raise SystemError('Only `None` (default initial condition) and '
                              '`nodelay` (skip initial condition) have been '
                              'implemented')
        # Create empty buffer array. N = size of window X # cols
        b = np.zeros((n, cols))

        # print("bshape is: ", b.shape)
        # Fill buffer by column handling for initial condition and overlap
        j = 0
        for i in range(cols):
            # Set first column to n values from x, move to next iteration
            if i == 0 and opt == 'nodelay':
                b[0:n, i] = x[0:n]
                continue
            # set first values of row to last p values
            elif i != 0 and p != 0:
                b[:p, i] = b[-p:, i-1]
            # If initial condition, set p elements in buffer array to zero
            else:
                b[:p, i] = 0
            # Assign values to buffer array from x
            b[p:, i] = x[p*(i+1):p*(i+2)]

        return b
