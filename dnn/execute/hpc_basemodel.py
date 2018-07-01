from dnn.base.constants.config import Config
from dnn.base.utils.log_error import initialize_logger
from dnn.base.utils.data_structures_utils import NumpyEncoder
import numpy as np
import json
import io
import os
try:
    to_unicode = unicode
except NameError:
    to_unicode = str

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
    
''' Class wrappers for writing HPC mvar model computations '''
class BaseHPC(object):
    ''' Required attribbutes when running a job array '''
    numcores = None 
    numgpus = None

    @staticmethod
    def get_available_gpus():
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

    def __init__(self, config=None):
        self.config = config or Config()
        self.logger = initialize_logger(
            self.__class__.__name__,
            self.config.out.FOLDER_LOGS)

    def loaddata(self):
        msg = "Base HPC model method is not implemented."
        raise NotImplementedError(msg)

    def createmodel(self):
        msg = "Base HPC create model method is not implemented."
        raise NotImplementedError(msg)
    
    def trainmodel(self, model, num_epochs, batch_size, 
                        train_dataset, test_dataset, 
                        outputdir, expname, device=None):
        msg = "Base HPC train model method is not implemented."
        raise NotImplementedError(msg)

    def testmodel(self, trainer, modelname):
        msg = "Base HPC test model method is not implemented."
        raise NotImplementedError(msg)

    def loadmetafile(self, metafilename):
        self.metadata = self._loadjsonfile(metafilename)

    def loadmetadata(self, metadata):
        self.metadata = metadata

    def _loadnpzfile(self, npzfilename):
        if not npzfilename.endswith('.npz'):
            npzfilename += '.npz'

        result = np.load(npzfilename)
        self.logger.debug("loaded npzfilename: {} with keys: {}".format(npzfilename, result.keys()))
        return result 

    def _writejsonfile(self, metadata, metafilename):
        if not metafilename.endswith('.json'):
            metafilename += '.json'
        with io.open(metafilename, mode='w', encoding='utf8') as outfile:
            str_ = json.dumps(metadata, 
                                indent=4, sort_keys=True, 
                                cls=NumpyEncoder, separators=(',', ': '), 
                                ensure_ascii=False)
            outfile.write(to_unicode(str_))

    def _loadjsonfile(self, metafilename):
        if not metafilename.endswith('.json'):
            metafilename += '.json'
        try:
            with open(metafilename, mode='r', encoding='utf8') as f:
                metadata = json.load(f)
            metadata = json.loads(metadata)
        except:
            with io.open(metafilename, encoding='utf-8', mode='r') as fp:
                json_str = fp.read()
            metadata = json.loads(json_str)
        return metadata