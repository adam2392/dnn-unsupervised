import numpy as np 
from dnn.base.constants.config import Config, OutputConfig
from dnn.base.utils.log_error import initialize_logger
import dnn.base.constants.model_constants as constants

from dnn.keras_models.metrics.classifier import BinaryClassifierMetric
from dnn.base.utils.data_structures_utils import NumpyEncoder
import os
import json
import pickle
import io
try:
    to_unicode = unicode
except NameError:
    to_unicode = str
class TrainMetrics(object):
    # metrics
    loss_queue = []
    recall_queue = []
    precision_queue = []
    accuracy_queue = []
    fp_queue = []


class TestMetrics(object):
    # metrics
    loss_queue = []
    recall_queue = []
    precision_queue = []
    accuracy_queue = []
    fp_queue = []


class BaseTrainer(object):
    metric_comp = BinaryClassifierMetric()
    model = None 
    def __init__(self, model, config=None):
        self.config = config or Config()
        self.logger = initialize_logger(
            self.__class__.__name__,
            self.config.out.FOLDER_LOGS)
        self.model = model

    def configure(self):
        msg = "Base trainer configure method is not implemented."
        raise NotImplementedError(msg)

    def train(self):
        msg = "Base trainer train method is not implemented."
        raise NotImplementedError(msg)

    def _summarize(self, outputs, labels, loss, regularize=False):
        pass

    def savemetricsoutput(self, modelname):
        metricsfilepath = os.path.join(self.outputdatadir, modelname+ "_metrics.json")
        auc = self.metrichistory.aucs
        fpr = self.metrichistory.fpr
        tpr = self.metrichistory.tpr 
        thresholds = self.metrichistory.thresholds
        metricdata = {
            'auc': auc,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
        }
        self._writejsonfile(metricdata, metricsfilepath)

    def saveoutput(self, modelname):
        modeljson_filepath = os.path.join(self.outputdatadir, modelname + "_model.json")
        history_filepath = os.path.join(
            self.outputdatadir,  modelname + '_history' + '.pkl')
        finalweights_filepath = os.path.join(
            self.outputdatadir, modelname + '_final_weights' + '.h5')
        self._saveoutput(modeljson_filepath, history_filepath, finalweights_filepath)

    def _saveoutput(self, modeljson_filepath, history_filepath, finalweights_filepath):
        # save model
        if not os.path.exists(modeljson_filepath):
            # serialize model to JSON
            model_json = self.model.to_json()
            self._writejsonfile(model_json, modeljson_filepath)
            # with open(modeljson_filepath, "w") as json_file:
            #     json_file.write(model_json)
            print("Saved model to disk")

        # save history
        with open(history_filepath, 'wb') as file_pi:
            pickle.dump(self.HH.history, file_pi)
        print("saved history file!")
        
        # save final weights
        self.model.save(finalweights_filepath)
        print("saved final weights file!")

    def _writejsonfile(self, metadata, metafilename):
        with io.open(metafilename, 'w', encoding='utf8') as outfile:
            str_ = json.dumps(metadata,
                              indent=4, sort_keys=True, cls=NumpyEncoder,
                              separators=(',', ': '), ensure_ascii=False)
            outfile.write(to_unicode(str_))

    def _loadjsonfile(self, metafilename):
        if not metafilename.endswith('.json'):
            metafilename += '.json'

        try:
            with open(metafilename, mode='r', encoding='utf8') as f:
                metadata = json.load(f)
            metadata = json.loads(metadata)
        except Exception as e:
            print(e)
            print("can't open metafile: {}".format(metafilename))
            with io.open(metafilename, encoding='utf-8', mode='r') as fp:
                json_str = fp.read() #json.loads(
            metadata = json.loads(json_str)

        self.metadata = metadata
        return self.metadata

    def visualize_filters(self, epoch):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import numpy.ma as ma
        import matplotlib as mpl
        mpl.use('Agg')

        def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
            """Wrapper around pl.imshow"""
            if cmap is None:
                cmap = cm.jet
            if vmin is None:
                vmin = data.min()
            if vmax is None:
                vmax = data.max()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
            pl.colorbar(im, cax=cax)
        #    pl.savefig("featuremaps--{}".format(layer_num) + '.jpg')

        
        def make_mosaic(imgs, nrows, ncols, border=1):
            """
            Given a set of images with all the same shape, makes a
            mosaic with nrows and ncols
            """
            nimgs = imgs.shape[0]
            imshape = imgs.shape[1:]

            mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                                    ncols * imshape[1] + (ncols - 1) * border),
                                    dtype=np.float32)

            paddedh = imshape[0] + border
            paddedw = imshape[1] + border
            for i in range(nimgs):
                row = int(np.floor(i / ncols))
                col = i % ncols

                mosaic[row * paddedh:row * paddedh + imshape[0],
                       col * paddedw:col * paddedw + imshape[1]] = imgs[i]
            return mosaic


        # Visualize weights
        W=model.layers[8].get_weights()[0][:,:,0,:]
        W=np.swapaxes(W,0,2)
        W = np.squeeze(W)
        print("W shape : ", W.shape)

        pl.figure(figsize=(15, 15))
        pl.title('conv1 weights')
        nice_imshow(pl.gca(), make_mosaic(W, 8, 8), cmap=cm.binary)

                