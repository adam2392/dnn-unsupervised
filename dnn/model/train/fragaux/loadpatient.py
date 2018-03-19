import sys
import os
import numpy as np
import pandas as pd

import datetime
import time
from dateutil.relativedelta import relativedelta

import utils as futil

class LoadPat(object):
    '''
    A class describing a dataset that is within our framework. This object will help set up the
    data for computation and any meta data necessary to link the computations together.

    This assumes that data is stored in a specific format!
    datadir/
        <patient1>
        <patient2>
            - <p>_rawnpy.npy
            - <p>_annotations.csv
            - <p>_chans.csv
            - <p>_headers.csv

    This mainly is the format that is the output of the dataconversion.py file from EDF files.

    Will need to reformat for other type of EEG files.
    '''

    def __init__(self, patient, datadir, clinoutcome=None):
        self.patient = patient
        self.datadir = datadir
        self.rawdatafile = patient+'_rawnpy.npy'
        self.annotationsfile = patient+'_annotations.csv'
        self.channelsfile = patient+'_chans.csv'
        self.headersfile = patient+'_headers.csv'
        self.clinoutcome = clinoutcome
        self.reference = None
        # get relevant channel data
        self.patid, self.seizid = futil.splitpatient(patient)
        self.included_chans, self.onsetchans, self.clinresult = futil.returnindices(self.patid, self.seizid)

        self.included_chans = self.included_chans.astype(int)
        if self.onsetchans is not None:
            self.onsetchans = np.asarray([x.lower() for x in list(self.onsetchans)])

        # print("Loaded ", patient)
        self.loadchans_fromfile()
        self.loadheaders_fromfile()
        self.loadannotations_fromfile()

        if len(self.included_chans) == 0:
            self.included_chans = np.arange(0, len(self.chanlabels)).astype(int)

    def _metafilepath(self, filename):
        return os.path.join(self.datadir, self.patient, filename)

    def get_clindata(self):
        clindata = {
            'onset_time': self.onset_time,
            'offset_time': self.offset_time,
            'samplerate': self.samplerate,
            'reference': self.reference,
            'gender': self.gender,
            'age': self.age
        }
        return clindata
    def loadchans_fromfile(self):
        chansfile = self._metafilepath(self.channelsfile)
        # read in the csv file into pandas
        chanheaders = pd.read_csv(chansfile)
        # read in labels
        chanlabels = chanheaders['Labels']
        self.chanlabels = chanlabels.values

        # apply lower case transformation to keep things simple
         # return only included chans
        if self.included_chans.size > 0:
            self.chanlabels[self.included_chans] = np.asarray([x.lower() for x in list(self.chanlabels[self.included_chans])])
        # print("Loaded channels file data")

    def loadrawdata_fromfile(self, reference=None):
        rawdatafile = self._metafilepath(self.rawdatafile)
        # load numpy array data
        rawdata = np.load(rawdatafile)

        self.reference = 'monopolar'
        # if signals by channels -> transpose
        if rawdata.shape[0] > rawdata.shape[1]:
            rawdata = rawdata.T

        # different montage/references
        if reference == 'avg': # perform average referencing
            # average over each row
            avg = np.mean(rawdata, axis=1, keepdims=True)
            rawdata = rawdata - avg
            self.reference = 'average'
        elif reference == 'bipolar': # perform bipolar montage
            self.reference = 'bipolar'
            # convert contacts into a list of tuples as data structure
            # contacts = []
            # for contact in chanlabels:
            #     thiscontactset = False
            #     for idx, s in enumerate(contact):
            #         if s.isdigit() and not thiscontactset:
            #             elec_label = contact[0:idx]
            #             thiscontactset = True
            #     contacts.append((elec_label, int(contact[len(elec_label):])))
            # # compute the bipolar scheme
            # recording = fragility.preprocess.seegrecording.SeegRecording(contacts, rawdata, samplefreq)
            # self.chanlabels = np.asarray(recording.get_channel_names_bipolar())
            # rawdata = recording.get_data_bipolar()

        # return only included chans
        if self.included_chans.size > 0:
            rawdata = rawdata[self.included_chans]
        # print("Loaded raw data")
        return rawdata
    def loadheaders_fromfile(self):
        headersfile = self._metafilepath(self.headersfile)
        # read in the file headers into pandas
        fileheaders = pd.read_csv(headersfile)
        
        # get important meta data (ADD ELEMENTS HERE TO ADD TO CLASS)
        birthdate = fileheaders['Birth Date']
        daterecording = fileheaders['Start Date (D-M-Y)']
        gender = fileheaders['Gender']
        equipment = fileheaders['Equipment']
        samplerate = fileheaders['Sample Frequency']
        recordduration = fileheaders['Data Record Duration (s)']
        
        # set metadata members of this class
        self.birthdate = birthdate.values[0]
        self.daterecording = daterecording.values[0]
        self.gender = gender.values[0]
        self.equipment = equipment.values[0]
        self.samplerate = samplerate.values[0]
        self.recordduration = recordduration.values[0]

        self.age = None
        # if not np.isnan(self.birthdate):
        try:
            now = datetime.datetime.now()
            birthdate = datetime.datetime.strptime(self.birthdate, '%d %b %Y')
            rdelta = relativedelta(now, birthdate)
            self.age = rdelta.years
            # print('Age in years - ', rdelta.years)
            # print('Age in months - ', rdelta.months)
            # print('Age in days - ', rdelta.days)
            # print("Loaded headers from file.")
        except TypeError:
            sys.stderr.write("type error in birth date probably, so just setting to nan")
        except ValueError:
            sys.stderr.write("value error in birth date probably, so just setting to nan")
    def loadannotations_fromfile(self):
        '''
        Here, we mainly are interested in the onset/offset times
        if they are present...
        '''
        annotationsfile = self._metafilepath(self.annotationsfile)
        # read in the clinical annotations into pandas
        annotations = pd.read_csv(annotationsfile)
        descriptions = annotations['Description'].values

        onset_time = []
        offset_time = []
        onsetset = False
        offsetset = False
        for idx, descrip in enumerate(descriptions):
            descrip = descrip.lower().split(' ')
            # print(descrip)
            if 'onset' in descrip \
                    or 'crise' in descrip \
                    or 'cgtc' in descrip \
                    or 'sz' in descrip or 'absence' in descrip:
                if not onsetset:
                    onset_time = annotations['Time (sec)'].values[idx]
                    onsetset = True
            if 'offset' in descrip \
                    or 'fin' in descrip \
                    or 'end' in descrip:
                if not offsetset:
                    offset_time = annotations['Time (sec)'].values[idx]
                    offsetset = True
        
        # set onset/offset times and markers
        # if np.array(onset_time).size > 0:
        try:
            self.onset_sec = onset_time
            self.onset_time = np.multiply(onset_time, 1000)
        except TypeError:
            print("no onset time!")
            self.onset_time = None
            self.onset_sec = None
        try:
            self.offset_sec = offset_time
            self.offset_time = np.multiply(offset_time, 1000)
        except TypeError:
            print("no offset time!")
            self.offset_time = None
            self.onset_sec
        # print('Ran setup of annotations data!')

    def readseiztimes(self, master_file):
        '''
        Helper function to read seizure times from a master excel file

        Mainly used to help in reading data from UMMC since it wasn't easily logged
        on the edf file.
        '''
        xl_file = pd.ExcelFile(master_file)
        center = 'UMMC'

        # load in the master excel sheet by a certain center
        dfs = {sheet_name: xl_file.parse(center) for sheet_name in xl_file.sheet_names}
        center_df = dfs[center]

        # get the patient row data
        center_df = center_df.apply(lambda x: x.astype(str).str.lower())
        pat_df = center_df.loc[center_df['Identifier'] == self.patient]

        convert_totime = lambda df_val: datetime.datetime.strptime(df_val, '%H:%M:%S')

        # get the onset/offset electrographic time
        record_start = convert_totime(pat_df['Recording Start'].item())
        onset_time = (convert_totime(pat_df['E Onset'].item()) - record_start).total_seconds()
        offset_time = (convert_totime(pat_df['E Offset'].item()) - record_start).total_seconds()
        self.onset_time = np.multiply(onset_time, self.samplerate)
        self.offset_time = np.multiply(offset_time, self.samplerate)
