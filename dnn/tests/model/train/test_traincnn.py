import pytest
from model.train.traincnn import TrainCNN

@pytest_fixture
def loadtrainer():
	trainer = TrainCNN()

class Test_TrainCNN():
	def test_saveoutput(self):
		modelname = 'test'
		outputdatadir = '~/Downloads/'

		trainer.saveoutput(modelname, outputdatadir)
		assert AttributeError

	def test_loadtrainingdata_vars(self):
		# pass in list of Xmain_train

		# pass in np.ndarray of X_train
		pass

	def test_loadtestingdata_vars(self):
		# pass in list of Xmain_test

		# pass in np.ndarray of X_test
		pass