# dnn-unsupervised
By: Adam Li
### Languages/Libs: Python, Ipython, Keras, Tensorflow, Pandas, SciPy
### Concepts: CNN, RNN, LSTM, CNN-LSTM, Mixing Networks, GPU Training

# TODO:
1. Adding logging statements throughout the package
- warn
- debug
- error
- info
- critical 
2. Make visualization library for models:
- 

# Citation (tbd):

# Background
IEEG data is sparse within the community relative to the big data of commercial sectors that power deep learning models. This project builds deep learning models that learn off of simulated data from nonlinear computational models of the brain.

Visualizations will be performed with matplotlib and seaborn.

# Installation
Clone and create virtual environment.

    git clone https://github.com/adam2392/dnn_unsupervised.git
    python3 -m venv .venv

# Setup

    pip install -r requirements.txt
    ./build_singularity.sh

Note, this project assumes your data is processed elsewhere. It is mainly concerned with formatting data into a specific format and training fully connected, convolutional, recurrent neural networks. It mainly depends on:

* scipy, numpy for linear algebra, digital signal processing and matrix operations
* pandas for basic io
* keras/tensorflow for TF type models
* pytorch for pytorch models
* SINGULARITY for running models on a cluster to get your singularity image pulled down from shub

# Running Models
## 1. VGG-16 CNN

## 2. CNN-LSTM

## 3. CNN-LSTM Mixing Network

## 4. Hybrid Networks


# Contact
Message me if you have questions, or create an issue.
