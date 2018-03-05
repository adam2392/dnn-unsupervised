from abc import ABCMeta
import numpy as np 
import pprint
class BaseTrain(metaclass=ABCMeta):
    requiredAttributes = ['NUM_EPOCHS', 'batch_size', 'AUGMENT']

    def train(self, model, xtrain, ytrain, xtest, ytest, batch_size, epochs, AUGMENT):
        msg = "Base training method is not implemented."
        raise NotImplementedError(msg)

    def setoptimizer(self):
        msg = "Base training method is not implemented."
        raise NotImplementedError(msg)

    def loadgenerator(self, generator):
        msg = "Base loading generator method is not implemented."
        raise NotImplementedError(msg)

    def loaddata(self):
        msg = "Base training method is not implemented."
        raise NotImplementedError(msg)

    def saveoutput(self):
        msg = "Base training method is not implemented."
        raise NotImplementedError(msg)

    def testoutput(self):
        msg = "Base training method is not implemented."
        raise NotImplementedError(msg)

    def summaryinfo(self):
        summary = {
            'batch_size': self.batch_size,
            'epochs': self.NUM_EPOCHS,
            'augment': self.AUGMENT
        }
        pprint.pprint(summary)