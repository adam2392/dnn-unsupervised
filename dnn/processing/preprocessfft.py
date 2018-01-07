import sys
import os
sys.path.append('/Users/adam2392/Documents/tvb/')
sys.path.append('/Users/adam2392/Documents/tvb/_tvbdata/')
sys.path.append('/Users/adam2392/Documents/tvb/_tvblibrary/')
from tvb.simulator.lab import *
import numpy as np
import pandas as pd
import scipy
from sklearn import cluster

from sklearn.preprocessing import MinMaxScaler

import tvbsim
from runmainsim import *

'''
Module for preprocessing and organizing data computed by FFT on iEEG data
into a huge compressed data structure

Data will be stored as npz compressed file

matrix will be H x W x F x T (height x width x frequency band x time window),
T x H x W x F 
where height and width define a grid where power is projected.

It will depend on the number of parcellated regions, so for 84, it will be a 
12x7 image x4 frequency bands (alpha, beta, gamma, high freq)
'''

# get factors of the number of regions
def get_factors(x):
   # This function takes a number and prints the factors

    factors = []
    for i in range(1, x + 1):
        if x % i == 0:
            factors.append(i)
    return factors


class PreProcess():
	def __init__(self,freqbands=None):

		if not freqbands:
			# establish frequency bands
			freqbands = {
			    'lowfreq': [0, 16],
			    'midfreq': [16, 33],
			    'gamma': [33, 90],
			    'highgamma': [90, 501],
			}
		self.freqbands = freqbands

	def _computefreqindices(self, freqs, freqbands):
		freqbandindices = {}
		for band in freqbands:
		    lowerband = freqbands[band][0]
		    upperband = freqbands[band][1]
		    
		    # get indices where the freq bands are put in
		    freqbandindices[band] = np.where((freqs >= lowerband) & (freqs < upperband))
		    freqbandindices[band] = [freqbandindices[band][0][0], freqbandindices[band][0][-1]]
		return freqbandindices

	def compresspowermat(self,datapath):
		freqbands = self.freqbands

		powerbands = {}
	    print os.path.join(datapath)
	    data = np.load(os.path.join(datapath))
	    
	    # extract data from the numpy file
	    power = data['power']
	    freqs = data['freqs']
	    timepoints = data['timepoints']
	    metadata = data['metadata'].item()
	    metadata['freqbands'] = freqbands
	  
	  	# compute the freq indices for each band
	  	if idx==0:
	  		freqbandindices = self._computefreqindices(freqs,freqbands)

	    # compress data using frequency bands
	    for band in freqbandindices:
	        indices = freqbandindices[band]
	        # average between these two indices
		    powerbands[band] = np.mean(power[:,indices[0]:indices[1]+1,:], axis=1) #[np.newaxis,:,:]

		return powerbands

	def createfilelist(self,datafiles):
		'''
		Create an accompanying list of datafiles to save, so that it is a metadata
		list of all the data that is compressed into the final data structure
		'''
		for datafile in datafiles:
			data = np.load(os.path.join(datadir,datafile))
			metadata = data['metadata'].item()



			onsettimes = metadata['onsettimes']
			offsettimes = metadata['offsettimes']

		pass

	def loadmetadata(self,patient,datafile):
		data = np.load(os.path.join(datadir,datafile))
    
		# extract data from the numpy file
		power = data['power']
		freqs = data['freqs']
		timepoints = data['timepoints']
		metadata = data['metadata'].item()

		return metadata

	def loadtvbdata(self, patient, project_dir)
		# RECOMPUTE GAIN MATRIX USING THIS PATIENT
		project_dir = os.path.join('/Volumes/ADAM LI/pydata/metadata/', patient)
		use_subcort = True

		# load in the vertices, normals and areas of gain matrix
		verts, normals, areas, regmap = tvbsim.util.read_surf(project_dir, use_subcort)

		confile = os.path.join(metadatadir, patient, "connectivity.zip")
		sensorsfile = os.path.join(metadatadir, patient, "seeg.txt")
	
		###################### 1. Structural Connectivity ########################
		con = tvbsim.initializers.connectivity.initconn(confile)

		return con

	def projectpower_gain(self,con,metadata):
		seeg_xyz = metadata['seeg_xyz']
		seeg_labels = metadata['seeg_contacts']

		# get the ez regions
		ezregion = metadata['ez']
		ezindices = metadata['ezindices']
		# extract the seeg_xyz coords and the region centers
		region_centers = con.centres
		regions = con.region_labels

		# reshape the regions of 84 into a parcellated rectangular "image"
		# height = np.ce
		factors = get_factors(len(regions))
		height = factors[len(factors)/2]
		width = len(regions) / height

		# # shapes of a new region/region centers
		# new_region_centers = np.reshape(region_centers, (height, width, 3), order='C')
		# new_regions = np.reshape(regions, (height,width), order='C')

		# check seeg_xyz moved correctly - In early simulation data results, was not correct
		buff = seeg_xyz - region_centers[:, np.newaxis]
		buff = np.sqrt((buff**2).sum(axis=-1))
		test = np.where(buff==0)
		indice = test[1]

		modgain = tvbsim.util.gain_matrix_inv_square(verts, areas,
                            regmap, len(regions), seeg_xyz)
		modgain = modgain.T

		# map seeg activity -> epileptor source and create data structure
		for idx,band in enumerate(powerbands):
		    mapped_power_band = np.tensordot(modgain, powerbands[band], axes=([1],[0]))
		        
		    if idx==0:
		        mapped_power_bands = mapped_power_band.reshape(height, width, mapped_power_band.shape[1], 
		                                                     order='C')[np.newaxis,:,:,:]
		    else:
		        mapped_power_bands = np.append(mapped_power_bands, 
		                    mapped_power_band.reshape(height, width, mapped_power_band.shape[1], 
		                                                     order='C')[np.newaxis,:,:,:],
		                                      axis=0)
		    print powerbands[band].shape
		    
		# new condensed data structure is H x W x F x T, to concatenate more, add to T dimension
		mapped_power_bands = mapped_power_bands.swapaxes(0,2)
		return mapped_power_bands

	def projectpower_knn(self):
		# map seeg_xyz to 3 closest region_centers
		tree = scipy.spatial.KDTree(region_centers)
		seeg_near_indices = []

		seeg_counter = np.zeros(len(regions))

		mapped_power_bands = np.zeros((len(regions), len(freqbands), powerbands[band].shape[1]))

		for ichan in range(0, len(seeg_labels)):
		    near_regions = tree.query(seeg_xyz[ichan,:].squeeze(), k=3)
		    near_indices = near_regions[1]
		    
		    # go through each frequency band and map activity onto those near indices
		    for idx,band in enumerate(powerbands):
		        chanpower = powerbands[band][ichan,:]
		        mapped_power_bands[near_indices,idx,:] += chanpower.astype('float64')
		    
		    seeg_counter[near_indices] += 1
		    seeg_near_indices.append(near_indices)

		# get the average based on how many contributions of the seeg power was to this region
		mapped_power_bands = mapped_power_bands / seeg_counter[:,np.newaxis,np.newaxis]
		
		# reshape for the correct output
		mapped_power_bands = mapped_power_bands.reshape(height, width, len(freqbands), powerbands[band].shape[1], 
                                                     order='C')
		mapped_power_bands = mapped_power_bands.swapaxes(0,1)

		return mapped_power_bands


	def projectpower_invsquare(self):
		# map seeg_xyz to the rest of the regions from a factor of 
		dr = region_centers - seeg_xyz[:,np.newaxis] # computes distance along each axis
		ndr = np.sqrt((dr**2).sum(axis=-1)) # computes euclidean distance
		Vr = 1/(ndr**2) # fall off as a function of r^2

		inf_indices = np.where(np.isinf(Vr))
		small_indices = np.where(ndr <= 1)

		# can either set to 1, or the max that there currently is + some small epsilon
		# the problem with setting to 1 is that the signal might drown out everything else
		Vr[small_indices] = np.nanmax(np.ma.masked_invalid(Vr[:])) + np.nanmin(Vr[:])

		# normalize Vr with minmax
		scaler = MinMaxScaler(feature_range=(0, 1))
		Vr = scaler.fit_transform(Vr)

		# map seeg activity -> epileptor source and create data structure
		for idx,band in enumerate(powerbands):
		    mapped_power_band = np.tensordot(Vr, powerbands[band], axes=([1],[0]))
		    
		    # store the formatted power bands
		    if idx==0:
		        mapped_power_bands = mapped_power_band.reshape(height, width, mapped_power_band.shape[1], 
		                                                     order='C')[np.newaxis,:,:,:]
		    else:
		        mapped_power_bands = np.append(mapped_power_bands, 
		                    mapped_power_band.reshape(height, width, mapped_power_band.shape[1], 
		                                                     order='C')[np.newaxis,:,:,:],
		                                      axis=0)
		    print powerbands[band].shape
		    
		# new condensed data structure is H x W x F x T, to concatenate more, add to T dimension
		mapped_power_bands = mapped_power_bands.swapaxes(0,2)

		return mapped_power_bands

