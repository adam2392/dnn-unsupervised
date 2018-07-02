import keras.backend as K
from keras.layers import Conv2D
from keras.layers import Activation, Lambda
from keras.models import Input, Model
import keras.layers

from keras.layers import ReLU, BatchNormalization

from dnn.keras_models.nets.generic.base import BaseGenericNet

class Inception(BaseGenericNet):
    @classmethod
    def residualblock(self, x, idx, numfilters):
        '''
        Utility function to build up the inception modules for an
        inception style network that operates at different scales (e.g.
        1x1, 3x3, 5x5 convolutions)
        '''

        # create the towers that occur during each layer of diff scale convolutions
        tower_0 = Conv2D(numfilters, kernel_size=(1,1),
                        padding='same',
                        activation='relu')(x)
        tower_1 = Conv2D(numfilters, kernel_size=(1,1), 
                        padding='same', 
                        activation='relu')(x)
        tower_1 = Conv2D(numfilters, kernel_size=(3,3), 
                        padding='same',
                        activation='relu')(tower_1)
        tower_2 = Conv2D(numfilters, kernel_size=(1,1), 
                        padding='same', 
                        activation='relu')(x)
        tower_2 = Conv2D(numfilters, kernel_size=(5,5), 
                        padding='same', 
                        activation='relu')(tower_2)
        tower_3 = MaxPooling2D((3,3), strides=(1,1), 
                        padding='same')(x)
        tower_3 = Conv2D(numfilters, kernel_size=(1,1), 
                        padding='same', 
                        activation='relu')(tower_3)
        # concatenate the layers and flatten
        output = keras.layers.concatenate([tower_0, tower_1, tower_2, tower_3], axis=1)

        return output