class BaseFormat(object):
	'''
	A base formatter for deep neural networks.

	We will be able to make sure that data computations are invariant to the version of python/numpy.
	This can be done by making sure the metadata is saved according to unicode strings always.`
	'''
	def __init__(self):
		pass

    def decodebytes(self, metadata):
    	'''
		A method for decoding a metadata dictionary from bytes -> unicode.
		This type of conversion is needed when we have Python2 npz files being saved and then
		read in by the Python3.
    	'''
        def convert(data):
            if isinstance(data, bytes):  return data.decode('ascii')
            if isinstance(data, dict):   return dict(map(convert, data.items()))
            if isinstance(data, tuple):  return map(convert, data)
            return data
        # try:
        metadata = {k.decode("utf-8"): (v.decode("utf-8") if isinstance(v, bytes) else v) for k,v in metadata.items()}
        for key in metadata.keys():
            metadata[key] = convert(metadata[key])
        return metadata

    def loaddatafiles(self, fftdatadir)
        # get all datafiles for the fft maps
        # fftdatadir = '/Volumes/ADAM LI/pydata/output/outputfft/tng/'
        # Get ALL datafiles from all downstream files
        self.datafiles = []
        for root, dirs, files in os.walk(fftdatadir):
            for file in files:
                if '.DS' not in file:
                    self.datafiles.append(os.path.join(root, file))

	def renamefiles(self, project_dir):
		'''
		Function used for renaming the seeg.xyz and gain-inv-square files
		into txt files for easy reading by python.
		'''

        ####### Initialize files needed to 
        # convert seeg.xyz to seeg.txt file
        sensorsfile = os.path.join(project_dir, "seeg.xyz")
        newsensorsfile = os.path.join(project_dir, "seeg.txt")
        
        try:
            os.rename(sensorsfile, newsensorsfile)
        except:
            print("Already renamed seeg.xyz possibly!")

        # convert gain_inv-square.mat file into gain_inv-square.txt file
        gainmatfile = os.path.join(project_dir, "gain_inv-square.mat")
        newgainmatfile = os.path.join(project_dir, "gain_inv-square.txt")
        try:
            os.rename(gainmatfile, newgainmatfile)
        except:
            print("Already renamed gain_inv-square.mat possibly!")

	def formatdata(self):
		'''
		An abstract function for any child class to inherit and implement
		'''
		raise NotImplementedError('Each formatter for our deep nn needs a formatdata function!')