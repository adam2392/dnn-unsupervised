# coding=utf-8

import numpy as np

# Default model parameters
LEARNING_RATE = 1e-3
DROPOUT = True
SHUFFLE = True

# frequency bands
DALPHA = [0, 15]
BETA = [15, 30]
GAMMA = [30, 90]
HIGH = [90, 200]

# Dataset constants
TRAIN = 'TRAIN'
TEST = 'TEST'
VALIDATE = 'VALIDATE'

# how can we inject noise into our models?
WHITE_NOISE = "White"
COLORED_NOISE = "Colored"
NOISE_SEED = 42
