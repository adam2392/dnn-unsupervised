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

import processing
import processing.preprocessfft
from processing.util import DataHandler
import peakdetect

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

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

class FormatFFT(BaseFormat):
    def __init__(self, rawdatadir, metadatadir, outputdatadir):
        # establish frequency bands
        freqbands = {
                'dalpha':[0,15],
                'beta':[15,30],
                'gamma':[30,90],
                'high':[90,200],
            }
        postprocessfft = processing.preprocessfft.PreProcess(freqbands=freqbands)

        self.winsizems = 500
        self.stepsizems = 250
        self.typetransform = 'fourier'
        self.mtbandwidth = 4
        self.mtfreqs = []
        self.rawdatadir = rawdatadir
        self.metadatadir = metadatadir
        self.outputdatadir = outputdatadir
        if not os.path.exists(self.outputdatadir):
            os.makedirs(self.outputdatadir)

    def formatdata(self):
        # rawdatadir = '/Volumes/ADAM LI/pydata/convertedtng/'
        checkrawdata = lambda patient: os.path.join(self.rawdatadir, patient)

        # define the data handler 
        datahandler = DataHandler()
        pca = PCA(n_components=2)

        for idx, datafile in enumerate(self.datafiles):
            print(idx)
            # perform file identification
            dirname = os.path.dirname(datafile)
            filename = path_leaf(datafile)
            fileid = filename.split('_fftmodel')[0]
            patient = '_'.join(fileid.split('_')[0:2])
            
            # load in the data for this fft computation
            fftdata = np.load(datafile, encoding='bytes')
            power = fftdata['power']
            freqs = fftdata['freqs']
            timepoints = fftdata['timepoints']
            metadata = fftdata['metadata'].item()
            
            # extract the metadata needed
            metadata = decodebytes(metadata) 
            onset_times = metadata['onsettimes']
            offset_times = metadata['offsettimes']
            seeg_labels = metadata['chanlabels']
            seeg_xyz = metadata['seeg_xyz']
            samplerate = metadata['samplerate']
            
            # get indices of channels that we have seeg_xyz for
            power = np.abs(power)
            
            # get overlapping indices on seeg with xyz
            xyzinds = [i for i,x in enumerate(seeg_labels) if any(thing==x for thing in seeg_labels)]
            seeg_xyz = seeg_xyz[xyzinds,:]
            
            print("Patient is: ", patient)
            print("file id is: ", fileid)
            assert power.shape[0] == seeg_xyz.shape[0]
            assert power.shape[0] == len(seeg_labels)
            
            # postprocess fft into bins
            power = postprocessfft.binFrequencyValues(power, freqs)

            # project xyz data
            seeg_xyz = pca.fit_transform(seeg_xyz)
            
            # Tensor of size [samples, freqbands, W, H] containing generated images.
            image_tensor = datahandler.gen_images(seeg_xyz, power, 
                                    n_gridpoints=32, normalize=False, augment=False, 
                                    pca=False, std_mult=0., edgeless=False)
            
            # compute ylabels    
            ylabels = datahandler.computelabels(onset_times, offset_times, timepoints)
            # instantiate metadata hash table
            metadata = dict()
            metadata['chanlabels'] = seeg_labels
            metadata['seeg_xyz'] = seeg_xyz
            metadata['ylabels'] = ylabels
            metadata['samplerate'] = samplerate
            metadata['timepoints'] = timepoints
            
            # save image and meta data
            imagefilename = os.path.join(trainimagedir, filename.split('.npz')[0])
            print(image_tensor.shape)
            print('saved at ', imagefilename)
            np.savez_compressed(imagefilename, image_tensor=image_tensor, metadata=metadata)
            
