from keras.callbacks import Callback
import numpy as np

class ReducLr(Callback):
    def on_train_begin(self, logs={}):
        return
 
    # def on_train_end(self, logs={}):
    #     return
 
    # def on_epoch_begin(self, epoch, logs={}):
    #     return
 
    # def on_batch_begin(self, batch, logs={}):
    #     return
 
    # def on_batch_end(self, batch, logs={}):
    #     return

    def on_epoch_end(self, epoch, logs={}):
        return