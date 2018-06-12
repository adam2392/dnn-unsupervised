from abc import ABCMeta
import numpy as np
import pprint
import sklearn


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

    '''
    These two functions for directly loading in the test/train datasets
    '''
    def loadtrainingdata_vars(self, Xmain_train, y_train):
        y_train = np.array(y_train)[:,np.newaxis]
        # Xmain_train = np.concatenate(Xmain_train, axis=0)
        # Xmain_train = np.vstack(Xmain_train)[..., np.newaxis]

        print(y_train.shape)
        print(Xmain_train.shape)
        # load the ylabeled data 1 in 0th position is 0, 1 in 1st position is 1
        invert_y = 1 - y_train
        y_train = np.concatenate((invert_y, y_train), axis=1)
        # format the data correctly
        class_weight = sklearn.utils.compute_class_weight('balanced',
                                                          np.unique(
                                                              y_train).astype(int),
                                                          np.argmax(y_train, axis=1))
        self.X_train = Xmain_train
        self.y_train = y_train
        self.class_weight = class_weight

    def loadtestingdata_vars(self, Xmain_test, y_test):
        y_test = np.array(y_test)[:,np.newaxis]
        # Xmain_test = np.vstack(Xmain_test)[..., np.newaxis]
        # load the ylabeled data 1 in 0th position is 0, 1 in 1st position is 1
        invert_y = 1 - y_test
        y_test = np.concatenate((invert_y, y_test), axis=1)
        self.X_test = Xmain_test
        self.y_test = y_test