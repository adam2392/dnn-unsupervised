import pytest
from model.nets.ieegrnn import iEEGRNN 

NUMLSTMLAYERS = 5
NUMTIMEWINS = 100

@pytest_fixture()
def loadmodel():
	dnnmodel = iEEGRNN()

class test_iEEGRNN():
	def test__build_deeprnn(self):
		celltype = ''
		dnnmodel.buildmodel(celltype=celltype, output=True)
		assert ValueError()

		celltype = 'lstm'
		dnnmodel.buildmodel(celltype=celltype, output=True)

		celltype = 'rnn'
		dnnmodel.buildmodel(celltype=celltype, output=True)

		celltype = 'gru'
		dnnmodel.buildmodel(celltype=celltype, output=True)