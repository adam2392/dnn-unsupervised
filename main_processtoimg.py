import os
import numpy as np 

from dnn.processing.format import formatfft

def main(fftdatadir, metadatadir, outputdatadir, gainmatfile):
    formatter = formatfft.FormatFFT(fftdatadir, metadatadir, outputdatadir)

    # load in the gainmat file, regions file and get all datafiles to analyze
    formatter.loadgainmat(gainmatfile)
    formatter.getdatafiles()
    formatter.formatdata()

if __name__ == '__main__':
    patid = 'id001_ac'

    # the main data directories that there is raw, meta, and output data
    metadatadir = '/Volumes/ADAM LI/pydata/metadata/'
    rawdatadir = '/Volumes/ADAM LI/pydata/convertedtng/'
    fftdatadir = '/Volumes/ADAM LI/pydata/output_fft/tng/win500_step250/'
    outputdatadir = '/Volumes/ADAM LI/pydata/output_fft/asimages/regions/'

    gainmatfile = os.path.join(metadatadir, patid, 'gain_inv-square.txt')
    # run conversion
    main(rawdatadir, metadatadir, outputdatadirgainmatfile)
