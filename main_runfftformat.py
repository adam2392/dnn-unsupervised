'''
File used to create the images using FFT computations.

Needs to read in a directory of fft computations that store the 
power spectrum as chans x freqs x timewins.

This can be converted into an imsize x imsize x chans images that 
can be stored.
'''

from dnn.processing.format import formatfft


def main_formatfft(rawdatadir, metadatadir, outputdatadir):
    formatter = formatfft.FormatFFT(rawdatadir, metadatadir, outputdatadir)

    # run computation to format all the data as images
    formatter.formatdata()

    return 1


if __name__ == '__main__':
    rawdatadir = ''
    metadatadir = ''
    outputdatadir = ''

    main_formatfft(rawdatadir, metadatadir, outputdatadir)
