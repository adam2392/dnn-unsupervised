from .base import BaseNet
############################ NUM PROCESSING FXNS ############################
import numpy as np
############################ UTILITY FUNCTIONS ############################
import time
import math as m

############################ ANN FUNCTIONS ############################
######### import DNN frameworks #########
import tensorflow as tf
import kerasa

from keras import Model
# import high level optimizers, models and layers
from keras.models import Sequential, Model
from keras.layers import InputLayer

# for CNN
from keras.layers import Conv1D, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, MaxPooling1D
from keras.layers import AveragePooling1D, AveragePooling2D
# for general NN behavior
from keras.layers import Dense, Dropout, Flatten, LeakyReLU
from keras.layers import Input, Concatenate, Permute, Reshape, Merge

# utility functionality for keras
from keras.layers.embeddings import Embedding
import pprint

class CapsNet():
	pass