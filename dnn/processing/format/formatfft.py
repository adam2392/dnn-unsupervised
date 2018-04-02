import sys
sys.path.append('../')
sys.path.append('/Users/adam2392/Documents/fragility_analysis/')
import fragility
from datainterface.loadpatient import LoadPat

sys.path.append('/Users/adam2392/Documents/tvb/')
sys.path.append('/Users/adam2392/Documents/tvb/_tvbdata/')
sys.path.append('/Users/adam2392/Documents/tvb/_tvblibrary/')
# from tvb.simulator.lab import *
import tvbsim.util

# import processing
from ..preprocessfft import PreProcess
from ..util import DataHandler
from ... import peakdetect

import os
import numpy as np
import scipy
import scipy.io
import pandas as pd
import time

from natsort import natsorted
import ntpath
from sklearn.decomposition import PCA
from shutil import copyfile

from .base import BaseFormat
from .loadregions import readconnectivity


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


class FormatFFT(BaseFormat):
    def __init__(self, fftdatadir, rawdatadir, metadatadir, outputdatadir):
        # establish frequency bands
        freqbands = {
            'dalpha': [0, 15],
            'beta': [15, 30],
            'gamma': [30, 90],
            'high': [90, 200],
        }
        postprocessfft = PreProcess(
            freqbands=freqbands)

        self.winsizems = 500
        self.stepsizems = 250
        self.typetransform = 'fourier'
        self.mtbandwidth = 4
        self.mtfreqs = []
        self.rawdatadir = rawdatadir
        self.fftdatadir = fftdatadir
        self.metadatadir = metadatadir
        self.outputdatadir = outputdatadir
        if not os.path.exists(self.outputdatadir):
            os.makedirs(self.outputdatadir)

    # def gain_matrix_inv_square(self, region_labels, regmap, vertices, seeg_xyz):
    def gain_matrix_inv_square(self):
        '''
        Computes a gain matrix using an inverse square fall off (like a mean field model)
        Parameters
        ----------
        vertices             np.ndarray of floats of size n x 3, where n is the number of vertices
        areas                np.ndarray of floats of size n x 3
        region_mapping       np.ndarray of ints of size n
        nregions             int of the number of regions
        sensors              np.ndarray of floats of size m x 3, where m is the number of sensors

        Returns
        -------
        np.ndarray of size m x n
        '''
        pass
        nregions = self.conn.region_labels.shape[0]
        nverts = self.vertices.shape[0]
        nsens = self.seeg_xyz.shape[0]
        reg_map_mtx = np.zeros((nverts, nregions), dtype=int)
        for i, region in enumerate(self.regmap):
            if region >= 0:
                reg_map_mtx[i, region] = 1
        gain_mtx_vert = np.zeros((nsens, nverts))
        for sens_ind in range(nsens):
            a = self.seeg_xyz[sens_ind, :] - self.vertices
            na = np.sqrt(np.sum(a**2, axis=1))

            # original version
            gain_mtx_vert[sens_ind, :] = self.areas / (na**2)

            # To Do: Refactor to use a more physically accurate way to project source activity
            # adding a 1 in the denominator to softmax the gain matrix
            softmax_inds = np.where(na < 1)[0]
            if len(softmax_inds) > 0:
                # print("na was less than one, so softmaxing here at 1.")
                # epsilon = 1 - a
                # na = np.sqrt(np.sum(a**2, axis=1))
                gain_mtx_vert[sens_ind, softmax_inds] = self.areas[softmax_inds] / \
                    (1 + na[softmax_inds]**2)

        return gain_mtx_vert.dot(reg_map_mtx)

    def getoverlapdata(self, seeg_xyz, seeg_labels, datalabels):
        '''
        Pass in the seeg_xyz/labels from the raw data.

        The datalabels is then passed which are the data labels from
        the computed step (fft, fragility, etc.).

        Output:
        - seeginds (np.ndarray) of seeg indices to use in the
        seeg_xyz/labels of the seeg.txt files
        '''
        # get overlapping indices
        seeg_labels_list = list(seeg_labels)
        inter_labels = list(set(datalabels) & set(seeg_labels))

        seeginds = []
        for idx, label in enumerate(inter_labels):
            seeginds.append(seeg_labels_list.index(label))
        seeginds.sort()
        seeginds = np.array(seeginds).astype(int)
        return seeginds
    
    def getsupersetinds(self, seeg_xyz, seeg_labels, datalabels):
        '''
        Pass in the seeg_xyz/labels from the raw data.

        The datalabels is then passed which are the data labels from
        the computed step (fft, fragility, etc.).

        Output:
        - seeginds (np.ndarray) of seeg indices to use in the
        seeg_xyz/labels of the seeg.txt files
        '''
        # get overlapping indices
        datalabels_list = list(datalabels)
        inter_labels = list(set(datalabels) & set(seeg_labels))

        seeginds = []
        for idx, label in enumerate(inter_labels):
            seeginds.append(datalabels_list.index(label))
        seeginds.sort()
        seeginds = np.array(seeginds).astype(int)
        return seeginds

    def loadgainmat(self, gainfile):
        # function to get model in its equilibrium value
        gain_pd = pd.read_csv(gainfile, header=None, delim_whitespace=True)
        self.gainmat = gain_pd.as_matrix().T

    def mapdatatoregs(self, data):
        # return np.tensordot(data,self.gainmat,axes=0)
        return np.matmul(self.gainmat, data)
        # or just compute the inverse from a least squares solution x = A \ b
        # return np.linalg.lstsq(data, gain_mat)

    def getdatafiles(self):
        datafiles = []

        for root, dirs, files in os.walk(self.fftdatadir):
            for file in files:
                if '.DS' not in file:
                    datafiles.append(os.path.join(root, file))
        self.datafiles = datafiles

    def loadregionsxyz(self, confile):
        con = readconnectivity(confile)
        regions_xyz = con['centres']
        self.regions_xyz = regions_xyz
    def loadseegxyz(self, seegfile):
        '''
        This is just a wrapper function to retrieve the seeg coordinate data in a pd dataframe
        '''
        seeg_pd = pd.read_csv(
            seegfile, names=['x', 'y', 'z'], delim_whitespace=True)
        self.seegfile = seegfile
        self.seeg_labels = seeg_pd.index.values
        self.seeg_xyz = seeg_pd.as_matrix(columns=None)

    def convertdatafromfile(self, datafile):
        # define the data handler
        datahandler = DataHandler()
        pca = PCA(n_components=2)

        # load in the data for this fft computation
        fftdata = np.load(datafile, encoding='bytes')
        print(fftdata.keys())
        power = fftdata['power']
        freqs = fftdata['freqs']
        timepoints = fftdata['timepoints']

        # only used if simulation
        metadata = fftdata['metadata'].item()
        # extract the metadata needed - only avail in tng sim data
        metadata = self.decodebytes(metadata)
        onset_times = metadata['onsettimes']
        offset_times = metadata['offsettimes']
        seeg_labels = metadata['chanlabels']
        seeg_xyz = metadata['seeg_xyz']
        samplerate = metadata['samplerate']

        # get indices of channels that we have seeg_xyz for
        power = np.abs(power)

        # get overlapping indices on seeg with xyz
        xyzinds = [i for i, x in enumerate(seeg_labels) if any(
            thing == x for thing in seeg_labels)]
        seeg_xyz = seeg_xyz[xyzinds, :]

        assert power.shape[0] == seeg_xyz.shape[0]
        assert power.shape[0] == len(seeg_labels)

        # postprocess fft into bins
        postprocessfft = PreProcess()
        power = postprocessfft.binFrequencyValues(power, freqs)

        # project xyz data
        seeg_xyz = pca.fit_transform(seeg_xyz)

        # Tensor of size [samples, freqbands, W, H] containing generated images.
        image_tensor = datahandler.gen_images(seeg_xyz, power,
                                              n_gridpoints=32, normalize=False, augment=False,
                                              pca=False, std_mult=0., edgeless=False)\
        # compute ylabels
        ylabels = datahandler.computelabels(
            onset_times, offset_times, timepoints)
        # instantiate metadata hash table
        metadata = dict()
        metadata['chanlabels'] = seeg_labels
        metadata['seeg_xyz'] = seeg_xyz
        metadata['ylabels'] = ylabels
        metadata['samplerate'] = samplerate
        metadata['timepoints'] = timepoints
        return image_tensor, metadata

    def converttoregs(self, datafile, dataloader=None):
        # define the data handler
        datahandler = DataHandler()

        # load in the data for this fft computation
        fftdata = np.load(datafile, encoding='bytes')
        print(fftdata.keys())
        power = fftdata['power']
        freqs = fftdata['freqs']
        timepoints = fftdata['timepoints']

        # load in the region centers and seeg xyz
        reg_xyz = self.regions_xyz
        seeg_xyz = self.seeg_xyz
        seeg_labels = self.seeg_labels

        # metadata = fftdata['metadata'].item()
        # extract the metadata needed - only avail in tng sim data
        # metadata = self.decodebytes(metadata)
        # onset_times = metadata['onsettimes']
        # offset_times = metadata['offsettimes']
        # seeg_labels = metadata['chanlabels']
        # samplerate = metadata['samplerate']
        # seeg_xyz = metadata['seeg_xyz']

        # get indices of channels that we have seeg_xyz for
        power = np.abs(power)

        onset_times = dataloader.onset_time
        offset_times = dataloader.offset_time
        trimmedchanlabels = dataloader.chanlabels
        samplerate = dataloader.samplerate

        seeginds = self.getoverlapdata(seeg_xyz, seeg_labels, trimmedchanlabels)
        # get the subset of the xyz/labels for seeg contacts
        seeg_xyz = seeg_xyz[seeginds, :]
        seeg_labels = seeg_labels[seeginds]
        self.gainmat = self.gainmat[:,seeginds]

        powerinds = self.getsupersetinds(seeg_xyz, seeg_labels, trimmedchanlabels)
        power = power[powerinds,...]

        assert power.shape[0] == len(seeg_labels)

        # postprocess fft into bins
        if samplerate <= 500:
            # establish frequency bands
            freqbands = {
                'dalpha': [0, 15],
                'beta': [15, 30],
                'gamma': [30, 90],
                'high': [90, 200],
            }
        else:
            freqbands = None
        postprocessfft = PreProcess(freqbands=freqbands)
        print(power.shape)
        power = postprocessfft.binFrequencyValues(power, freqs)

        # Tensor of size [samples, freqbands, W, H] containing generated images.
        image_tensor = np.zeros((reg_xyz.shape[0], power.shape[1], power.shape[2]))
        for isamp in range(power.shape[2]):
            image_tensor[...,isamp] = self.mapdatatoregs(power[...,isamp].squeeze())

        # compute ylabels
        ylabels = datahandler.computelabels(
            onset_times, offset_times, timepoints)

        print('Converttoregs summary of data:')
        print("power shape: ", power.shape)
        print("regions xyz shape: ",reg_xyz.shape)
        print("gainmat shape: ", self.gainmat.shape)
        print("image tensor shape: ", image_tensor.shape)
        print("Y labels shape: ", ylabels.shape)
        # instantiate metadata hash table
        metadata = dict()
        metadata['chanlabels'] = seeg_labels
        metadata['seeg_xyz'] = seeg_xyz
        metadata['reg_xyz'] = reg_xyz
        metadata['ylabels'] = ylabels
        metadata['samplerate'] = samplerate
        metadata['timepoints'] = timepoints
        return image_tensor, metadata

    def formatdata(self):
        # rawdatadir = '/Volumes/ADAM LI/pydata/convertedtng/'
        def checkrawdata(patient): return os.path.join(
            self.rawdatadir, patient)

        for idx, datafile in enumerate(self.datafiles):
            print(idx)
            # perform file identification
            dirname = os.path.dirname(datafile)
            filename = path_leaf(datafile)
            fileid = filename.split('_fftmodel')[0]
            patient = '_'.join(fileid.split('_')[0:2])

            # get the connectivity and gainmatrix files
            confile = os.path.join(self.metadatadir, patient, 'connectivity.zip')
            gainmatfile = os.path.join(self.metadatadir, patient, 'gain_inv-square.txt')
            seegfile = os.path.join(self.metadatadir, patient, 'seeg.txt')
            
            # load in the onset/offset times
            dataloader = LoadPat(patient=fileid, datadir=self.rawdatadir)

            self.loadgainmat(gainmatfile)
            self.loadregionsxyz(confile)
            self.loadseegxyz(seegfile)
            print("Patient is: ", patient)
            print("file id is: ", fileid)

            # convert into an image tensor and metadata
            # image_tensor, metadata = self.convertdatafromfile(datafile)
            image_tensor, metadata = self.converttoregs(datafile, dataloader)

            # save image and meta data
            imagefilename = os.path.join(
                self.outputdatadir, filename.split('.npz')[0])
            print(image_tensor.shape)
            print('saved at ', imagefilename)
            np.savez_compressed(
                imagefilename, image_tensor=image_tensor, metadata=metadata)
