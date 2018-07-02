import keras.backend as K
from keras.layers import Conv2D
from keras.layers import Activation, Lambda
from keras.models import Input, Model
import keras.layers

from keras.layers import ReLU, BatchNormalization

from dnn.keras_models.nets.generic.base import BaseGenericNet

class VGG(BaseGenericNet):
    def __init__(self, length_imsize, width_imsize, n_colors):
        self.length_imsize = length_imsize
        self.width_imsize = width_imsize
        self.n_colors = n_colors

    # @classmethod
    def residualblock(self, x, idx, ilay,
                    numfilters, 
                    kernel_size,
                    kernel_init):
        original_x = x

        conv = Conv2D(numfilters*(2 ** idx),
                        kernel_size=kernel_size,
                        kernel_initializer=kernel_init,
                        input_shape=(self.length_imsize, 
                                    self.width_imsize, 
                                    self.n_colors),
                        # dilation_rate=dilation_rate,
                        activation='linear',
                        use_bias=False,
                        name='vgg_conv_{}_relu_s{}'.format(idx, ilay))(x)

        x = BatchNormalization(axis=-1, momentum=0.99, 
                epsilon=0.001, center=True, scale=True, 
                beta_initializer='zeros', gamma_initializer='ones', 
                moving_mean_initializer='zeros', moving_variance_initializer='ones', 
                beta_regularizer=None, gamma_regularizer=None, 
                beta_constraint=None, gamma_constraint=None)(conv)
        # x = LeakyReLU(alpha=0.1)(x)
        x = Activation('relu')(x)
        return x
