import os
import numpy as np 
import pandas as pd 
import mne
import json
import sys
from .utils import utils
from .utils import seegrecording
from .readers.read_connectivity import LoadConn

from fragility.base.constants.config import Config
from fragility.base.utils.log_error import initialize_logger
from datetime import date
import warnings

warnings.filterwarnings("ignore", ".*not conform to MNE naming conventions.*")

class LoadDataset(object):
    def __init__(self, root_dir, datafile, patient=None, preload=False, reference='monopolar', config=None):
        self.config = config or Config()
        self.logger = initialize_logger(self.__class__.__name__, self.config.out.FOLDER_LOGS)

        self.root_dir = root_dir
        self.datafile = datafile
        self.reference = reference
        self.patient = patient 
        
        # set directories for the datasets 
        self.seegdir = os.path.join(self.root_dir, 'seeg', 'fif')
        self.elecdir = os.path.join(self.root_dir, 'elec')
        self.dwidir = os.path.join(self.root_dir, 'dwi')
        self.tvbdir = os.path.join(self.root_dir, 'tvb')
        
        if preload:
            self.loadmetadata()
            self.loadrawdata()
    def load_data(self):
        self.loadmetadata()
        self.loadrawdata()

    def loadmetadata(self):
        # sync good data         
        # self.sync_good_data()
        self.logger.info('Reading in metadata!')
        # rename files from .xyz -> .txt
        self._renamefiles()
        self._loadseegxyz()
        self._mapcontacts_toregs()
        # load in ez hypothesis and connectivity from TVB pipeline
        self._loadezhypothesis()
        self._loadconnectivity()

    def loadrawdata(self):
        # run main loader
        self._loadfile()
        self._loadinfodata()
        # get relevant channel data - ONLY USED FOR NON TNG PIPELINE DATA
        # if patient is not None:
        #     self.patid, self.seizid = utils.splitpatient(patient)
        #     self.included_chans, self.onsetchans, self.clinresult = utils.returnindices(
        #         self.patid, self.seizid)

        #     # mask bad channels - HARDCODED included chans...
        #     self.maskbadchans(self.included_chans)             
        # apply referencing to data and channel labels (e.g. monopolar, average, bipolar)
        self.referencedata(FILTER=False)

    def _renamefiles(self):
        sensorsfile = os.path.join(self.elecdir, 'seeg.xyz')
        newsensorsfile = os.path.join(self.elecdir, 'seeg.txt')
        try:
            # copyfile(sensorsfile, newsensorsfile)
            os.rename(sensorsfile, newsensorsfile)
        except:
            self.logger.info("\nAlready renamed seeg.xyz possibly!\n")

    def _loadseegxyz(self):
        seegfile = os.path.join(self.elecdir, 'seeg.txt')
        seeg_pd = utils.loadseegxyz(seegfile)
        self.chanxyz_labels = seeg_pd.index.values
        self.chanxyz = seeg_pd.as_matrix(columns=None)

        self.logger.info("\nLoaded in seeg xyz coords!\n")

    def _mapcontacts_toregs(self):
        contacts_file = os.path.join(self.elecdir, 'seeg.txt')
        label_volume_file = os.path.join(self.dwidir, 'label_in_T1.nii.gz')
        if not os.path.exists(label_volume_file):
            label_volume_file = os.path.join(self.dwidir, 'label_in_T1.dk.nii.gz')
        self.contact_regs = utils.mapcontacts_toregs(contacts_file, label_volume_file)

        self.logger.info("\nMapped contacts to regions!\n")

    def _loadmetadata(self, metafile):
        if not metafile.endswith('.json'):
            metafile += '.json'

        metafile = open(metafile)
        metajson = metafile.read()
        self.metadata = json.loads(metajson)
        self.logger.info("\nLoaded in metadata!\n")

    def _loadezhypothesis(self):
        ez_file = os.path.join(self.tvbdir, 'ez_hypothesis.txt')
        if not os.path.exists(ez_file):
            ez_file = os.path.join(self.tvbdir, 'ez_hypothesis.dk.txt')

        reginds = pd.read_csv(ez_file, delimiter='\n').as_matrix()
        self.ezinds = np.where(reginds==1)[0]
        self.logger.info("\nLoaded in ez hypothesis!\n")

    def _loadconnectivity(self):
        tvb_sourcefile = os.path.join(self.tvbdir, 'connectivity.zip')
        if not os.path.exists(tvb_sourcefile):
            tvb_sourcefile = os.path.join(self.tvbdir, 'connectivity.dk.zip')
            
        conn_loader = LoadConn()
        conn = conn_loader.readconnectivity(tvb_sourcefile)
        self.conn = conn
        self.logger.info("\nLoaded in connectivity!\n")

    def _loadfile(self):
        metadatafilepath = os.path.join(self.seegdir, self.datafile)
        sys.stderr.write("The meta data file to use is %s \n" % metadatafilepath)
        self._loadmetadata(metadatafilepath)
        # load in the useful metadata
        rawfile = self.metadata['filename']
        
        # set if this is a tngpipeline dataset
        datafilepath = os.path.join(self.seegdir, rawfile)

        # extract raw object
        if datafilepath.endswith('.edf'):
            raw = mne.io.read_raw_edf(datafilepath, 
                                        # preload=True,
                                        verbose=False)
        elif datafilepath.endswith('.fif'):
            raw = mne.io.read_raw_fif(datafilepath, 
                                        # preload=True,
                                        verbose=False)
        else:
            sys.stderr.write("Is this a real dataset? %s \n" % datafilepath)
            print("Is this a real dataset? ", datafilepath)

        # provide loader object access to raw mne object
        self.raw = raw

        # get events
        if datafilepath.endswith('.edf'):
            events = self.raw.find_edf_events()
            self.events = np.array(events)
            self.__setevents(self.events)
        # else:
        #     events = mne.find_events(raw,
        #                             stim_channel=[])
        #     self.events = np.array(events)
        # if edf file and one with events...
        # self.__setevents(self.events)

    def _loadinfodata(self):
        # set samplerate
        self.samplerate = self.raw.info['sfreq']
        # set channel names
        self.chanlabels = self.raw.info['ch_names']
        # also set to all the channel labels
        self.allchans = self.chanlabels
        self._processchanlabels()
        # set edge freqs that were used in recording
        # Note: the highpass_freq is the max frequency we should be able to see then.
        self.lowpass_freq = self.raw.info['lowpass']
        self.highpass_freq = self.raw.info['highpass']
        # set recording date
        record_date = date.fromtimestamp(self.raw.info["meas_date"][0])
        self.record_date = record_date
        record_ms_date = self.raw.info["meas_date"][1] # number of microseconds
        self.record_ms_date = record_ms_date

        # set line freq
        self.linefreq = self.raw.info['line_freq']
        if self.linefreq is None:
            self.linefreq = 50
            # self.linefreq = 60
            self.logger.info("\nHARD SETTING THE LINE FREQ. MAKE SURE TO CHANGE BETWEEN USA/FRANCE DATA!\n")  
  
        # else grab it from the json object
        self.onset_sec = self.metadata['onset']
        self.offset_sec = self.metadata['termination']
        badchans = self.metadata['bad_channels']
        try:
            nonchans = self.metadata['non_seeg_channels']
        except:
            nonchans = []
        self.badchans = badchans + nonchans 
        self.sztype = self.metadata['type']
        if self.offset_sec is not None:
            self.offset_ind = np.multiply(self.offset_sec, self.samplerate)
            self.onset_ind = np.multiply(self.onset_sec, self.samplerate)
        else:
            self.offset_ind = None
            self.onset_ind = None      
    
    def _processchanlabels(self):
        self.chanlabels = [str(x).replace('POL', '').replace(' ', '')
                      for x in self.chanlabels]

    def __setevents(self, events):
        eventonsets = events[:,0]
        eventdurations = events[:,1]
        eventnames = events[:,2]

        # initialize list of onset/offset seconds
        onset_secs = []
        offset_secs = []
        onsetset = False
        offsetset = False

        # iterate through the events and assign onset/offset times if avail.
        for idx, name in enumerate(eventnames):
            name = name.lower().split(' ')
            if 'onset' in name \
                    or 'crise' in name \
                    or 'cgtc' in name \
                    or 'sz' in name or 'absence' in name:
                if not onsetset:
                    onset_secs = eventonsets[idx]
                    onsetset = True
            if 'offset' in name \
                    or 'fin' in name \
                    or 'end' in name:
                if not offsetset:
                    offset_secs = eventonsets[idx]
                    offsetset = True

        # set onset/offset times and markers
        try:
            self.onset_sec = onset_secs
            self.onset_ind = np.ceil(onset_secs * self.samplerate)
        except TypeError:
            self.logger.info("no onset time!")
            self.onset_sec = None
            self.onset_ind = None
        try:
            self.offset_sec = offset_secs
            self.offset_ind = np.ceil(offset_secs * self.samplerate)
        except TypeError:
            self.logger.info("no offset time!")
            self.offset_sec = None
            self.offset_ind = None

    def sync_good_data(self, rawdata=None):
        if rawdata is None:
            # rawdata = self.rawdata
            self.rawdata, self.times = self.raw.get_data()

        badchans = self.badchans
        chanxyz_labs = self.chanxyz_labels
        chanlabels = self.chanlabels
        contact_regs = self.contact_regs
        chanxyz = self.chanxyz

        # map badchans, chanlabels to lower case
        badchans = np.array([lab.lower() for lab in badchans])
        chanxyz_labs = np.array([lab.lower() for lab in chanxyz_labs])
        chanlabels = np.array([lab.lower() for lab in chanlabels])

        # extract necessary metadata
        goodchans_inds = [idx for idx,chan in enumerate(chanlabels) if chan not in badchans if chan in chanxyz_labs]

        # only grab the good channels specified
        goodchan_labels = chanlabels[goodchans_inds]
        # rawdata = rawdata[goodchans_inds,:]

        # now sift through our contacts with xyz coords and region_mappings
        reggoodchans_inds = [idx for idx,chan in enumerate(chanxyz_labs) if chan in goodchan_labels]
        contact_regs = contact_regs[reggoodchans_inds]
        chanxyz = chanxyz[reggoodchans_inds,:]              
        
        # covert to numpy arrays
        contact_regs = np.array(contact_regs)
        goodchan_labels = np.array(goodchan_labels)

        # reject white matter contacts
        graychans_inds = np.where(np.asarray(contact_regs) != -1)[0]
        self.contact_regs = contact_regs[graychans_inds]
        self.rawdata = rawdata[graychans_inds,:]
        self.chanxyz = chanxyz[graychans_inds,:]
        self.goodchan_labels = goodchan_labels[graychans_inds]

        # set the indices needed here.
        # self.goodchans = goodchans_inds
        # self.goodchans2 = reggoodchans_inds
        # self.graychans_inds = graychans_inds

        print(self.contact_regs.shape)
        print(self.goodchan_labels.shape)
        print(self.rawdata.shape)
        print(self.chanxyz.shape)

        assert self.contact_regs.shape[0] == self.goodchan_labels.shape[0]
        assert self.goodchan_labels.shape[0] == self.rawdata.shape[0]
        assert self.rawdata.shape[0] == self.chanxyz.shape[0]

    def filter_data(self, rawdata=None):
        # the bandpass range to pass initial filtering
        freqrange =  [0.5]              
        freqrange.append(self.samplerate//2 - 1)
        # the notch filter to apply at line freqs
        linefreq = int(self.linefreq)           # LINE NOISE OF HZ
        assert linefreq == 50 or linefreq == 60
        sys.stderr.write("Line freq is: %s" % linefreq)
        # initialize the line freq and its harmonics
        freqs = np.arange(linefreq,251,linefreq)
        freqs = np.delete(freqs, np.where(freqs>self.samplerate//2)[0])
        # sys.stderr.write("Going to filter at freqrange: \n")
        # sys.stderr.write(freqrange)

        if rawdata is None:
            # apply band pass filter
            self.raw.filter(l_freq=freqrange[0],
                            h_freq=freqrange[1])
            # apply the notch filter
            self.raw.notch_filter(freqs=freqs)
        else:
            # print('Filtering!', freqrange)
            rawdata = mne.filter.filter_data(rawdata,
                                            sfreq=self.samplerate,
                                            l_freq=freqrange[0],
                                            h_freq=freqrange[1],
                                            # pad='reflect',
                                            verbose=False
                                            )
            rawdata = mne.filter.notch_filter(rawdata,
                                            Fs=self.samplerate,
                                            freqs=freqs,
                                            verbose=False
                                            )
            return rawdata
    def referencedata(self, FILTER=False):
        # Should we apply bandpass filtering and notch of line noise?
        if FILTER:
            self.filter_data()

        # apply referencing to data and channel labels (e.g. monopolar, average, bipolar)
        self.raw = self.raw.load_data()

        if self.reference == 'average':
            self.logger.info('\nUsing average referencing!\n')
            self.raw.set_eeg_reference(ref_channels="average", projection=False)
        elif self.reference == 'monopolar':
            self.logger.info('\nUsing monopolar referencing!\n')
            self.raw.set_eeg_reference(ref_channels=[], projection=False)
        elif self.reference == 'bipolar':
            self.logger.info("\nUsing bipolar referencing!\n")
            self.logger.debug("NEED TO CALL ALL PREPROCESSING FUNCTIONS")

            assert 1 == 0
            # convert contacts into a list of tuples as data structure
            contacts = []
            for contact in self.chanlabels:
                thiscontactset = False
                for idx, s in enumerate(contact):
                    if s.isdigit() and not thiscontactset:
                        elec_label = contact[0:idx]
                        thiscontactset = True
                contacts.append((elec_label, int(contact[len(elec_label):])))
            
            self.rawdata, self.times = self.raw.get_data()
            # compute the bipolar scheme
            recording = util.seegrecording.SeegRecording(
                contacts, self.rawdata, self.samplerate)
            self.chanlabels = np.asarray(recording.get_channel_names_bipolar())
            self.rawdata = recording.get_data_bipolar()

    def maskbadchans(self, included_chans=None):
        if self.reference == 'bipolar':
            warnings.warn('Bipolar referencing could not work with hard coded included chans!')
        
        if included_chans is None:
            warnings.warn('Included chans is hardcoded as: NONE')
            self.logger.info('Doing nothing in maskbadchans')
        else:
            # apply mask over the data
            self.chanlabels = self.chanlabels[included_chans]
            self.rawdata = self.rawdata[included_chans]

    def getmetadata(self):
        """
        If the user wants to clip the data, then you can save a separate metadata
        file that contains all useful metadata about the dataset.
        """
        metadata = dict()
        # Set data from the mne file object
        metadata['samplerate'] = self.samplerate
        metadata['chanlabels'] = self.chanlabels
        metadata['lowpass_freq'] = self.lowpass_freq
        metadata['highpass_freq'] = self.highpass_freq
        metadata['record_date'] = self.record_date
        metadata['onsetsec'] = self.onset_sec
        metadata['offsetsec'] = self.offset_sec
        metadata['reference'] = self.reference
        metadata['linefreq'] = self.linefreq
        metadata['onsetind'] = self.onset_ind
        metadata['offsetind'] = self.offset_ind
        metadata['allchans'] = self.allchans
        metadata['rawfilename'] = self.datafile 
        metadata['patient'] = self.patient

        # Set data from external text, connectivity, elec files
        metadata['ez_region'] = self.conn.region_labels[self.ezinds]
        metadata['region_labels'] = self.conn.region_labels
        # metadata['ez_chans'] = self.contact_regs == metadata['ez_region']
        metadata['chanxyz'] = self.chanxyz
        metadata['contact_regs'] = self.contact_regs
        
        try:
            metadata['chunklist'] = self.winlist
            metadata['secsperchunk'] = self.secsperchunk
        except:
            self.logger.info('chunking not set for %s %s \n' % (self.patient, self.datafile))

        # needs to be gotten from sync_data()
        # metadata['goodchans'] = dataloader.goodchans
        # metadata['graychans'] = dataloader.graychans_inds
        try:
            metadata['included_chans'] = self.included_chans
        except:
            self.logger.info('included_chans not set for %s %s \n' % (self.patient, self.datafile))

        return metadata

    def computechunks(self, secsperchunk=60):
        """
        Function to compute the chunks through the data by intervals of 60 seconds.

        This can be useful for sifting through the data one range at time.

        Note: The resolution for any long-term frequency analysis would be 1/60 Hz, 
        which is very low, and can be assumed to be DC anyways when we bandpass filter.
        """
        self.secsperchunk = secsperchunk
        samplerate = self.samplerate
        numsignalsperwin = np.ceil(secsperchunk*samplerate).astype(int)

        numsamps = self.raw.n_times
        winlist = []

        # define a lambda function to subtract window
        winlen = lambda x: x[1] - x[0]
        for win in self._chunks(np.arange(0,numsamps),numsignalsperwin):
            # ensure that each window length is at least numsignalsperwin
            if winlen(win) < numsignalsperwin - 1 and winlist:
                winlist[-1][1] = win[1]
            else:
                winlist.append(win)

        # push last win to cover the sample
        # lastwin = winlist[-1]
        # if lastwin[1] != numsamps - 1:
        #     winlist[-1][1] = numsamps - 1
        self.winlist = winlist

    def _chunks(self, l, n):
        """ 
        Yield successive n-sized chunks from l.
        """
        for i in range(0, len(l), n):
            chunk = l[i:i+n]
            yield [chunk[0], chunk[-1]]
            
    def clipdata_chunks(self, ind=None):
        """
        Function to clip the data. It could either returns:
             a generator through the data, or 
             just returns data at that index through the index

        See code below.

        """
        # if ind is None:
        #     # produce a generator that goes through the window list
        #     for win in self.winlist:
        #         data, times = self.raw[:,win[0]:win[-1]+1]
        #         yield data, times
        # else:
        
        win = self.winlist[ind]
        data, times = self.raw[:,win[0]:win[1]+1]
        return data,times

