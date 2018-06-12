from abc import ABCMeta
import numpy as np
import pprint


class BaseTrain(metaclass=ABCMeta):
    requiredAttributes = ['model', 'X_test', 'y_test']

    def loadmodel(self, modelfile):
        msg = "Base testing method is not implemented."
        raise NotImplementedError(msg)

    def saveoutput(self):
        msg = "Base testing method is not implemented."
        raise NotImplementedError(msg)

    def testoutput(self):
        msg = "Base testing method is not implemented."
        raise NotImplementedError(msg)

    def summaryinfo(self):
        summary = {
            'model': self.model,
            'X_test': self.X_test,
            'y_test': self.y_test
        }
        return summary
