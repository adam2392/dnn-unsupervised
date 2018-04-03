import numpy as np
import os
import sys
from sklearn.decomposition import PCA

sys.path.append('/Users/adam2392/Documents/fragility_analysis/')
sys.path.append('/home-1/ali39@jhu.edu/work/fragility_analysis')
from .loadpatient import LoadPat
import sklearn
from sklearn.model_selection import train_test_split, LeaveOneOut, LeaveOneGroupOut

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def compute_fragilitymetric(minnormpertmat):
    # get dimensions of the pert matrix
    N, T = minnormpertmat.shape
    # assert N < T
    fragilitymat = np.zeros((N, T))
    for icol in range(T):
        fragilitymat[:, icol] = (np.max(minnormpertmat[:, icol]) - minnormpertmat[:, icol]) /\
            np.max(minnormpertmat[:, icol])
    return fragilitymat


def getpatient(datafile):
    strsplits = datafile.split('/')
    patient = strsplits[-2]
    return patient


def getpatdata(patient, rawdatadir):
    # testing bipolar montage for raw data from clinical
    dataloader = LoadPat(patient, rawdatadir)
    onsetchans = dataloader.onsetchans
    onsettimes = dataloader.onset_time
    offsettimes = dataloader.offset_time
    samplerate = dataloader.samplerate

    # get the chan labels
    chanlabels = dataloader.chanlabels

class LabelData(object):
    '''
    Class for labeling the entire dataset based on what clinicians
    thought were the clinical EZ set.


    '''
    def __init__(self, numwins, rawdatadir):
        self.numwins = numwins
        self.rawdatadir = rawdatadir

    def loaddirofdata(self, datadir, listofpats):
        self.datafilepaths = []
        for root, dirs, files in os.walk(datadir):
            for file in files:
                # datafilepaths.append(os.path.join(root, file))
                # if any(pat in file for pat in patstoignore):

                # adding clause to ignore la01_ictal_2 for now:
                if any(pat in file for pat in listofpats) and 'inter' not in file \
                        and 'aw' not in file \
                        and 'aslp' not in file \
                        and 'la01_ictal_2' not in file:
                    # print('ignoring ', file)
                    self.datafilepaths.append(os.path.join(root, file))
                else:
                    # datafilepaths.append(os.path.join(root,file))
                    print('ignoring ', file)
    def decodebytes(self, metadata):
        def convert(data):
            if isinstance(data, bytes):
                return data.decode('ascii')
            if isinstance(data, dict):
                return dict(map(convert, data.items()))
            if isinstance(data, tuple):
                return map(convert, data)
            return data
        # try:
        metadata = {k.decode("utf-8"): (v.decode("utf-8")
                                        if isinstance(v, bytes) else v) for k, v in metadata.items()}
        for key in metadata.keys():
            metadata[key] = convert(metadata[key])
        return metadata

    def getinds(self, chanlabels, ezchans):
        # get onset channels - and their indices
        allinds = np.arange(0, len(chanlabels)).astype(int)
        # get the ez indices
        ezinds = []
        for ezchan in ezchans:
            try:
                buff = np.min(np.where(chanlabels == ezchan))
                ezinds.append(buff)
            except ValueError:
                # buff = np.argmax(chanlabels == ezchan)
                # if buff:
                #     buff = buff[0]
                buff = None
        if not ezinds:
            print('no ezinds here...')
        return allinds, ezinds

    def slicetimewins(self, timepoints, onsettimes, offsettimes):
        inds = []
        beginwin = 0
        endwin = 0
        onsetind = None
        offsetind = None

        # get the indices that the times occur at
        if onsettimes:
            onsetind = self._converttimestowindow(timepoints, onsettimes)
            inds.append(onsetind)
        if offsettimes:
            offsetind = self._converttimestowindow(timepoints, offsettimes)
            inds.append(offsetind)

        # get the number of wins
        if len(inds) == 2:
            currnumwins = offsetind - onsetind
            remwins = self.numwins - currnumwins

            beginwin = onsetind - remwins / 2
            endwin = offsetind + remwins / 2
        elif onsetind:
            beginwin = onsetind - self.numwins/2
            endwin = onsetind + self.numwins/2
        elif offsetind:
            beginwin = offsetind - self.numwins/2
            endwin = offsetind + self.numwins/2
        else:  # inds is empty
            beginwin = 0
            endwin = self.numwins

        return int(beginwin), int(endwin)
    def _converttimestowindow(self, timepoints, time):
        timeindex = np.where(np.logical_and(timepoints[:,0] < time, 
                                            timepoints[:,1] > time))[0][0]
        return timeindex
    def formatdata(self):
        '''
        Function used to take the list of auxiliary and main
        inputs, and reformat them into sequences of images with
        aux/main inputs.

        Then creates a list of ylabels as you go through these 
        aux/main inputs.
        '''
        rawdatadir = self.rawdatadir

        # load all the data
        main_list = []  # (samp, 1, timewins)
        ylabels = []   # (samp, 1)
        listofpats = []  # keep track of the list of patients

        for datafile in self.datafilepaths:
            datanpz = np.load(datafile, encoding='bytes')
            metadata = datanpz['metadata'].item()
            metadata = self.decodebytes(metadata)
            timepoints = metadata['timepoints']

            # get the patient data and chan labels
            patient = getpatient(datafile)

            dataloader = LoadPat(patient, rawdatadir)
            chanlabels = dataloader.chanlabels[dataloader.included_chans]
            ezchans = dataloader.onsetchans
            # get the onset/offset times
            onsettimes = dataloader.onset_time
            offsettimes = dataloader.offset_time

            # compute the fragility matrix
            fragmat = compute_fragilitymetric(datanpz['pertmats']).T
            assert fragmat.shape[0] > fragmat.shape[1]
            # downsize the storage of data
            fragmat = fragmat.astype('float32')

            allinds, ezinds = self.getinds(chanlabels, ezchans)

            # loop through all channels and separate into datasets
            
            for ichan in range(0, len(allinds)):
                fragmat_main = fragmat[:, ichan].T

                # get the begin/ending window and slice the fragility mats
                beginwin, endwin = self.slicetimewins(
                    timepoints, onsettimes, offsettimes)
                fragmat_main = fragmat_main[beginwin:endwin]

                assert fragmat_main.shape[0] == self.numwins

                # add to the list
                main_list.append(fragmat_main)

                if ichan in ezinds:
                    ylabels.append(1)
                else:
                    ylabels.append(0)

                # keep track of the list of pats used
                listofpats.append(patient)

        # set these to be accessed by other class methods
        self.listofpats = listofpats
        self.ylabels = ylabels
        self.main_data = main_list

    def trainingscheme(self, scheme='rand'):
        listofpats = np.array(self.listofpats)
        main_data = np.array(self.main_data)
        ylabels = np.array(self.ylabels)

        def _loo():
            '''
            Perform leave one out train/test split with the groups of patients
            we have.
            '''
            # initialize lists for the training/testing data sets
            X_train = []
            X_test = []
            y_train = []
            y_test = []

            logo = LeaveOneGroupOut()
            print('in leave one group out.')
            print(np.array(main_data).shape)
            print(np.array(ylabels).shape)
            print(np.array(listofpats).shape)
            for train_index, test_index in logo.split(X=main_data, y=ylabels, groups=listofpats):
                X_train.append(main_data[train_index,:])
                X_test.append(main_data[test_index,:])
                y_train.append(ylabels[train_index])
                y_test.append(ylabels[test_index])

            X_train = np.vstack(X_train)[..., np.newaxis]
            X_test = np.vstack(X_test)[..., np.newaxis]
            y_train = np.array(y_train)
            y_test = np.array(y_test)

            print('Finished setting up data')
            print(X_train.shape)
            print(X_test.shape)
            print(y_train.shape)
            print(y_test.shape)

            return X_train, X_test, y_train, y_test

        def _randsplit():
            # format the data correctly
            Xmain_train, Xmain_test,\
                y_train, y_test = train_test_split(
                    main_data, ylabels, test_size=0.33, random_state=42)
            return Xmain_train, Xmain_test, y_train, y_test

        schemes = ['loo', 'rand']
        if scheme not in schemes:
            raise ValueError('Scheme needs to be either rand, or loo')
        if scheme == 'loo':
            Xmain_train, Xmain_test, y_train, y_test = _loo()
        elif scheme == 'rand':
            Xmain_train, Xmain_test, y_train, y_test = _randsplit()

        # compute the class weights, if we need them
        class_weight = sklearn.utils.compute_class_weight('balanced',
                                                          np.unique(
                                                              ylabels).astype(int),
                                                          ylabels)
        self.class_weight = class_weight
        self.Xmain_train = Xmain_train
        self.y_train = y_train
        self.Xmain_test = Xmain_test
        self.y_test = y_test

    def transformdata(data, numwins):
        '''
        Function to transform the data before input. We may meed to downsample, or pad
        the data to get our desired 'length'.

        '''
        if data.shape[1] > numwins:
            # just simply slice off the end of the data
            data = data[:, 0:numwins]
        else:
            # need to apply padding
            # data =
            print('padding')


class SplitData(LabelData):
    '''
    Subclass the Label data class to include a way to get 
    an auxiliary data class (e.g. all other channels)
    '''
    def __init__(self, imsize, numwins, rawdatadir):
        super(SplitData, self).__init__(numwins, rawdatadir)
        self.imsize = imsize  # pc components

    def helperpca(self, data, numcomp):
        pca = PCA(n_components=numcomp)
        return pca.fit_transform(data)

    def formatdata(self):
        '''
        Function used to take the list of auxiliary and main
        inputs, and reformat them into sequences of images with
        aux/main inputs.

        Then creates a list of ylabels as you go through these 
        aux/main inputs.
        '''
        rawdatadir = self.rawdatadir

        # load all the data
        aux_list = []  # (samp, PC, timewins)
        main_list = []  # (samp, 1, timewins)
        ylabels = []   # (samp, 1)
        listofpats = []  # keep track of the list of patients

        for datafile in self.datafilepaths:
            datanpz = np.load(datafile, encoding='bytes')
            metadata = datanpz['metadata'].item()
            metadata = self.decodebytes(metadata)
            timepoints = metadata['timepoints']

            # get the patient data and chan labels
            patient = getpatient(datafile)

            dataloader = LoadPat(patient, rawdatadir)
            chanlabels = dataloader.chanlabels[dataloader.included_chans]
            ezchans = dataloader.onsetchans
            # get the onset/offset times
            onsettimes = dataloader.onset_time
            offsettimes = dataloader.offset_time

            # compute the fragility matrix
            fragmat = compute_fragilitymetric(datanpz['pertmats']).T
            assert fragmat.shape[0] > fragmat.shape[1]
            # downsize the storage of data
            fragmat = fragmat.astype('float32')

            allinds, ezinds = self.getinds(chanlabels, ezchans)

            print(datafile)
            # print(len(allinds))
            # print(len(chanlabels))
            # print(ezchans)
            # print(ezinds)

            # loop through all channels and separate into datasets
            for ichan in range(0, len(allinds)):
                # get the other indices
                otherinds = [i for i in range(len(allinds)) if i != ichan]

                fragmat_main = fragmat[:, ichan]
                fragmat_aux = fragmat[:, otherinds]

                # PCA Feature Space - for the channels
                # apply pca to the fragmat, and get the main fragility mat
                fragmat_aux = self.helperpca(fragmat_aux, numcomp=self.imsize)
                fragmat_aux = fragmat_aux.T
                fragmat_main = fragmat_main.T

                # print(fragmat_aux.shape)
                # print(fragmat_main.shape)

                # get the begin/ending window and slice the fragility mats
                beginwin, endwin = self.slicetimewins(
                    timepoints, onsettimes, offsettimes)
                # print(beginwin, endwin)
                fragmat_aux = fragmat_aux[:, beginwin:endwin]
                fragmat_main = fragmat_main[beginwin:endwin]

                assert fragmat_aux.shape[0] == self.imsize
                assert fragmat_aux.shape[1] == self.numwins
                assert fragmat_main.shape[0] == self.numwins

                # add to the list
                aux_list.append(fragmat_aux)
                main_list.append(fragmat_main)

                if ichan in ezinds:
                    ylabels.append(1)
                else:
                    ylabels.append(0)

                # keep track of the list of pats used
                listofpats.append(patient)

                # return aux_list, main_list
                # print(fragmat_main.shape)
                # print(fragmat_aux.shape)
                # print(fragmat.shape)
                # break
        # set these to be accessed by other class methods
        self.listofpats = listofpats
        self.ylabels = ylabels
        self.aux_data = aux_list
        self.main_data = main_list

    def trainingscheme(self, scheme='rand'):
        aux_data = self.aux_data
        main_data = self.main_data
        ylabels = self.ylabels

        def loo():
            # format the data correctly
            Xmain_train, Xmain_test,\
                Xaux_train, Xaux_test,\
                y_train, y_test = LeaveOneOut(aux_data, main_data, ylabels)
            return Xmain_train, Xmain_test, Xaux_train, Xaux_test, y_train, y_test

        def randsplit():
            # format the data correctly
            Xmain_train, Xmain_test,\
                Xaux_train, Xaux_test,\
                y_train, y_test = train_test_split(
                    aux_data, main_data, ylabels, test_size=0.33, random_state=42)
            return Xmain_train, Xmain_test, Xaux_train, Xaux_test, y_train, y_test

        schemes = ['loo', 'rand']
        if scheme not in schemes:
            raise ValueError('Scheme needs to be either rand, or loo')
        if scheme == 'loo':
            Xmain_train, Xmain_test, Xaux_train, Xaux_test, y_train, y_test = loo()
        elif scheme == 'rand':
            Xmain_train, Xmain_test, Xaux_train, Xaux_test, y_train, y_test = randsplit()

        # compute the class weights, if we need them
        class_weight = sklearn.utils.compute_class_weight('balanced',
                                                          np.unique(
                                                              ylabels).astype(int),
                                                          ylabels)
        self.class_weight = class_weight
        self.Xmain_train = Xmain_train
        self.Xaux_train = Xaux_train
        self.y_train = y_train

        self.Xmain_test = Xmain_test
        self.Xaux_test = Xaux_test
        self.y_test = y_test

    def transformdata(data, numwins):
        '''
        Function to transform the data before input. We may meed to downsample, or pad
        the data to get our desired 'length'.

        '''
        if data.shape[1] > numwins:
            # just simply slice off the end of the data
            data = data[:, 0:numwins]
        else:
            # need to apply padding
            # data =
            print('padding')
