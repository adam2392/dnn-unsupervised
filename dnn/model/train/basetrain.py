from abc import ABCMeta
import numpy as np
import pprint


class BaseTrain(metaclass=ABCMeta):
    requiredAttributes = ['dnnmodel', 'NUM_EPOCHS', 'batch_size', 'AUGMENT']

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

    def saveoutput(self, modelname, outputdatadir):
        modeljsonfile = os.path.join(outputdatadir, modelname + "_model.json")
        historyfile = os.path.join(
            outputdatadir,  modelname + '_history' + '.pkl')
        finalweightsfile = os.path.join(
            outputdatadir, modelname + '_final_weights' + '.h5')

        # save model
        if not os.path.exists(modeljsonfile):
            # serialize model to JSON
            model_json = self.dnnmodel.to_json()
            with open(modeljsonfile, "w") as json_file:
                json_file.write(model_json)
            print("Saved model to disk")

        # save history
        with open(historyfile, 'wb') as file_pi:
            pickle.dump(self.HH.history, file_pi)

        # save final weights
        self.dnnmodel.save(finalweightsfile)