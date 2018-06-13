# coding=utf-8
import numpy as np

# Default model parameters
LEARNING_RATE = 1e-3
DROPOUT = True
SHUFFLE = True
AUGMENT = True

NUM_EPOCHS = 100 # 200 # 300
BATCH_SIZE = 64 # 32

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
NOISE_SEED = 42
