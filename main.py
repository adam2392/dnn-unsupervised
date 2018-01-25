import sys
sys.path.append('./dnn/')

import model.ieeg_cnn_rnn
import model.train

import processing.util as util

import keras
from keras.callbacks import ModelCheckpoint
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale


def poly_decay(epoch, NUM_EPOCHS, INIT_LR):
    # initialize the maximum number of epochs, base learning rate,
    # and power of the polynomial
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0
    # compute the new learning rate based on polynomial decay
    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
    # return the new learning rate
    return alpha

if __name__ == '__main__':
	traindatadir = str(sys.argv[1])
	tempdatadir = str(sys.argv[2])
	
	# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
	os.environ["CUDA_VISIBLE_DEVICES"]=cuda_dev

	##################### PARAMETERS FOR NN ####################
	# image parameters
	imsize=32
	numfreqs = 5

	# layer parameters
	w_init = None
	n_layers = (4,2,1)
	poolsize = (2,2)
	filtersize = (3,3)

	size_fc = 1024
	DROPOUT = False #True
	ieegdnn = model.ieeg_cnn_rnn.IEEGdnn(imsize=imsize, n_colors=numfreqs)

	##################### INPUT DATA FOR NN ####################
	# define data filepath to images
	image_filepath = os.path.join(traindatadir, 'trainimages.npy')
	images = np.load(image_filepath)

	# load the ylabeled data
	ylabel_filepath = os.path.join(traindatadir, 'trainlabels.npy')
	ylabels = np.load(ylabel_filepath)
	invert_y = 1 - ylabels
	ylabels = np.concatenate((ylabels, invert_y),axis=1)

	##################### TRAINING FOR NN ####################
	# checkpoint
	filepath=os.path.join(tempdatadir,"weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5")
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')


	# VGG-12 style later
	vggcnn = ieegdnn._build_2dcnn(w_init=w_init, n_layers=n_layers, 
	                              poolsize=poolsize, filter_size=filtersize)
	vggcnn = ieegdnn._build_seq_output(vggcnn, size_fc, DROPOUT)
	print(vggcnn.summary())

	# load in CNN/LSTM
	# num_timewins = 5
	# size_mem = 128
	# size_fc = 1024
	# DROPOUT = False
	# cnn_lstm = ieegdnn.build_cnn_lstm(num_timewins=num_timewins, 
	#                                   size_mem=size_mem, size_fc=size_fc, DROPOUT=DROPOUT)
	# print(cnn_lstm.get_shape)

	# split into train,valid,test sets
	datahandler = util.DataHandler()

	# images = scale(images, axis=0, with_mean=True, with_std=True, copy=True )

	# format the data correctly 
	# (X_train, y_train), (X_val, y_val), (X_test, y_test) = datahandler.reformatinput(images, labels)
	X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.33, random_state=42)
	X_train = X_train.astype("float32")
	X_test = X_test.astype("float32")

	loss = 'binary_crossentropy'
	optimizer = keras.optimizers.Adam(lr=0.001, 
	                                        beta_1=0.9, 
	                                        beta_2=0.999,
	                                        epsilon=1e-08,
	                                        decay=0.0)
	metrics = ['accuracy']
	cnn_config = ieegdnn.compile_model(vggcnn, loss=loss, optimizer=optimizer, metrics=metrics)


	NUM_EPOCHS = 100
	batch_size = 32 # or 64... or 24
	# construct the image generator for data augmentation and construct
	# the set of callbacks
	aug = keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1,
	    height_shift_range=0.1, horizontal_flip=True,
	    fill_mode="nearest")
	# callbacks_list = [checkpoint]
	callbacks = [checkpoint, poly_decay]
	INIT_LR = 5e-3
	G=1
	HH = vggcnn.fit_generator(
	    aug.flow(X_train, y_train, batch_size=batch_size * G), # adds augmentation to data using generator
	    validation_data=(X_test, y_test),  
	    steps_per_epoch=len(X_train) // (batch_size * G),    #
	    epochs=NUM_EPOCHS,
	    callbacks=callbacks, verbose=2)

	# save final history object
	vggcnn.save('final_weights.h5')


	# datagen = ImageDataGenerator(
 #        featurewise_center=True, # set input mean to 0 over the dataset
 #        samplewise_center=False, # set each sample mean to 0
 #        featurewise_std_normalization=True, # divide inputs by std of the dataset
 #        samplewise_std_normalization=False, # divide each input by its std
 #        zca_whitening=False, # apply ZCA whitening
 #        rotation_range=20, # randomly rotate images in the range (degrees, 0 to 180)
 #        width_shift_range=0.2, # randomly shift images horizontally (fraction of total width)
 #        height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
 #        horizontal_flip=True, # randomly flip images
 #        vertical_flip=False) # randomly flip images

	# datagen.fit(X_sample) # let's say X_sample is a small-ish but statistically representative sample of your data

	# # let's say you have an ImageNet generator that yields ~10k samples at a time.
	# for e in range(nb_epoch):
	#     print("epoch %d" % e)
	#     for X_train, Y_train in ImageNet(): # these are chunks of ~10k pictures
	#         for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size=32): # these are chunks of 32 samples
	#             loss = model.train(X_batch, Y_batch)

	# # Alternatively, without data augmentation / normalization:
	# for e in range(nb_epoch):
	#     print("epoch %d" % e)
	#     for X_train, Y_train in ImageNet(): # these are chunks of ~10k pictures
	#         model.fit(X_batch, Y_batch, batch_size=32, nb_epoch=1)


