import pickle
import json

class History(object):
    history = {
        "train": {
            'loss': [],
            'fp': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'epoch': []
        },
        "val": {
            'loss': [],
            'fp': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'epoch': []
        }
    }
    curr_epoch = 0

    def update(self, train_metrics, val_metrics):
        self.curr_epoch += 1

        for metric in train_metrics.keys():
            metricvalue = train_metrics[metric]

            if metric in self.history['train'].keys():
                self.history['train'][metric].append(metricvalue)
                self.history['train']['epoch'].append(self.curr_epoch)

        for metric in val_metrics.keys():
            metricvalue = val_metrics[metric]

            if metric in self.history['val'].keys():
                self.history['val'][metric].append(metricvalue)
                self.history['val']['epoch'].append(self.curr_epoch)

    def save(self, historyfile):
        # save history
        with open(historyfile, 'wb') as file_pi:
            pickle.dump(self.history, file_pi)

    def open(self, historyfile):
        with open(historyfile, "rb") as f:
            dump = pickle.load(f)
        return dump
            # Now you can use the dump object as the original one  
            # self.some_property = dump.some_property