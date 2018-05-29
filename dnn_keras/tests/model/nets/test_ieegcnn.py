import pytest
from model.nets.ieegcnn import iEEGCNN
import keras


NUMINCEPTIONLAYERS=10
SIZEFC=1024

@pytest_fixture()
def createmodel():
	dnnmodel = iEEGCNN()

class Test_iEEGCNN(object):
	'''
	Test for the CNN classes, where we have VGG-style
	networks, inception style networks, and resnet
	style networks.

	We want to test the functionality of the networks
	with our proposed inputs.

	User can change the global vars to test if their model
	will pass the tests
	'''
	imsize = 32
	imsize = 0
	imsize = None

	n_colors = 4
	n_colors = 0
	n_colors = None

	def test__build_inception_towers(self):
		'''
		Testing the inception tower build up for various
		filter sizes
		'''
		imsize = 32
		n_colors = 4
		num_filters = 32
		input_img = Input(shape=(imsize, 
								imsize, 
								n_colors))
		dnnmodel._build_inception_towers(input_img,
										num_filters)

		num_filters = 64
		input_img = Input(shape=(imsize, 
						imsize, 
						n_colors))
		dnnmodel._build_inception_towers(input_img,
										num_filters)

		num_filters = 128
		input_img = Input(shape=(imsize, 
						imsize, 
						n_colors))
		dnnmodel._build_inception_towers(input_img,
										num_filters)

	def test_build_inception2dcnn(self):
		num_layers = NUMINCEPTIONLAYERS
		size_fc = SIZEFC
		num_filters = 64

		dnnmodel.build_inception2dcnn(num_layers,
									num_filters,
									size_fc)

	def test_build_vgg_2dcnn(self):
		pass

	def test_build_vgg_1dcnn(self):
		pass