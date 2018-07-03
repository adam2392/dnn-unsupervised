import keras.backend as K
from keras.layers import Conv1D, SpatialDropout1D
from keras.layers import Activation, Lambda
from keras.models import Input, Model
import keras.layers

from dnn.keras_models.nets.generic.base import BaseGenericNet

def channel_normalization(x):
    # Normalize by the highest activation
    max_values = K.max(K.abs(x), 2, keepdims=True) + 1e-5
    out = x / max_values
    return out
        
class TCN(BaseGenericNet):
    @classmethod
    def residual_block(self, x, s, i, activation, nb_filters, kernel_size, kernel_init):
        original_x = x
        conv = Conv1D(filters=nb_filters, 
                    kernel_size=kernel_size,
                    kernel_initializer=kernel_init,
                    dilation_rate=2 ** i, 
                    padding='causal',
                    activation='linear',
                    name='dilated_conv_%d_relu_s%d' % (2 ** i, s))(x)
        if activation == 'norm_relu':
          x = Activation('relu')(conv)
          x = Lambda(channel_normalization)(x)
        else:
          x = Activation(activation)(conv)

        # add dropout across space
        # x = SpatialDropout1D(0.05)(x)

        # 1x1 conv.
        x = Conv1D(nb_filters, 1, padding='same')(x)
        res_x = keras.layers.add([original_x, x])
        return res_x, x

    @classmethod
    def residual_block_basic(self, x, s, i, activation, nb_filters, kernel_size, kernel_init):
        original_x = x
        conv = Conv1D(filters=nb_filters, 
                    kernel_size=kernel_size,
                    kernel_initializer=kernel_init,
                    padding='same',
                    activation='linear',
                    name='dilated_conv_%d_relu_s%d' % (2 ** i, s))(x)
        if activation == 'norm_relu':
          x = Activation('relu')(conv)
          x = Lambda(channel_normalization)(x)
        else:
          x = Activation(activation)(conv)

        # add dropout across space
        # x = SpatialDropout1D(0.05)(x)

        # 1x1 conv.
        x = Conv1D(nb_filters, 1, padding='same')(x)
        res_x = keras.layers.add([original_x, x])
        return res_x, x