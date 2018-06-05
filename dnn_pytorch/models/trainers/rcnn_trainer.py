import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, StepLR, ReduceLROnPlateau

from dnn_pytorch.models.trainers.base import BaseTrainer
from dnn_pytorch.util import utils
from dnn_pytorch.models.evaluate import evaluate
from dnn_pytorch.models.metrics.classifier import BinaryClassifierMetric
import dnn_pytorch.base.constants.model_constants as constants

from dnn_pytorch.base.constants.config import Config, OutputConfig
from dnn_pytorch.models.regularizer.post_class_regularizer import Postalarm
import tensorboardX  # import SummaryWriter
from tqdm import trange


class RCNNTrainer(BaseTrainer):
    metric_comp = BinaryClassifierMetric()
    post_regularizer = None

    break
    # create a dictionary of metrics with their corresponding "lambda"
    # functions
    metrics = {
        'accuracy': metric_comp._accuracy,
        'recall': metric_comp._recall,
        'precision': metric_comp._precision,
        'fp': metric_comp._fp
    }

    def __init__(self, net, num_epochs, batch_size,
                 # device(s) to train on
                 device=None,
                 testoutputdir=None,
                 expname=None,                         # for gen. experiment logging
                 learning_rate=constants.LEARNING_RATE,
                 dropout=constants.DROPOUT, shuffle=constants.SHUFFLE,
                 config=None):
        '''         SET LOGGING DIRECTORIES: MODEL, TENSORBOARD         '''
        self.expname = expname
        self.testoutputdir = testoutputdir
        super(CNNTrainer, self).__init__(net=net,
                                         device=device,
                                         config=config)

        # Hyper parameters - training
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.save_summary_steps = 10

        # hyper parameters - dataset
        self.shuffle = shuffle

        # set tensorboard writer
        self._setdirs()  # set directories for all logging
        self.writer = tensorboardX.SummaryWriter(self.tboardlogdir)

        self.logger.info(
            "Logging output data to: {}".format(
                self.outputdatadir))
        self.logger.info(
            "Logging experimental data at: {}".format(
                self.explogdir))
        self.logger.info(
            "Logging tensorboard data at: {}".format(
                self.tboardlogdir))

    def _setdirs(self):
        # set where to log outputs of explog
        if self.testoutputdir is None:
            self.explogdir = os.path.join(
                self.config.tboard.FOLDER_LOGS,
                self.expname,
                'traininglogs')
            self.tboardlogdir = os.path.join(
                self.config.tboard.FOLDER_LOGS, self.expname, 'tensorboard')
            self.outputdatadir = os.path.join(
                self.config.tboard.FOLDER_LOGS, self.expname, 'output')
        else:
            self.explogdir = os.path.join(self.testoutputdir, 'traininglogs')
            self.tboardlogdir = os.path.join(self.testoutputdir, 'tensorboard')
            self.outputdatadir = os.path.join(self.testoutputdir, 'output')

        if not os.path.exists(self.explogdir):
            os.makedirs(self.explogdir)
        if not os.path.exists(self.tboardlogdir):
            os.makedirs(self.tboardlogdir)
        if not os.path.exists(self.outputdatadir):
            os.makedirs(self.outputdatadir)

    def composedatasets(self, train_dataset_obj, test_dataset_obj):
        self.train_loader = DataLoader(train_dataset_obj,
                                       batch_size=self.batch_size,
                                       shuffle=self.shuffle,
                                       num_workers=1)
        self.test_loader = DataLoader(test_dataset_obj,
                                      batch_size=self.batch_size,
                                      shuffle=self.shuffle,
                                      num_workers=1)
        # get input characteristics
        self.imsize = train_dataset_obj.imsize
        self.n_colors = train_dataset_obj.n_colors
        # size of training/testing set
        self.train_size = len(train_dataset_obj)
        self.val_size = len(test_dataset_obj)

        self.logger.info(
            "Each training epoch is {} steps and each validation is {} steps.".format(
                self.train_size, self.val_size))
        self.logger.info(
            "Setting the datasets for training/testing in trainer object!")
        self.logger.info(
            "Image size is {} with {} colors".format(
                self.imsize, self.n_colors))

    def loadmetrics(self, y_true, y_pred, metricholder):
        self.metric_comp.compute_scores(y_true, y_pred)

        # add to list for the metrics
        # self.metricholder.recall_queue.append(self.metrics.recall)
        # self.metricholder.precision_queue.append(self.metrics.precision)
        # self.metricholder.fp_queue.append(self.metrics.fp)
        # self.metricholder.accuracy_queue.append(self.metrics.accuracy)

    def run_config(self):
        """
        Configuration function that can change:
        - sets optimizer
        - sets loss function
        - sets scheduler
        - sets post-prediction-regularizer
        """
        optimparams = {
            'lr': self.learning_rate,
            'amsgrad': True,
        }

        optimizer = torch.optim.Adam(self.net.parameters(),
                                     **optimparams)

        # optim.SGD([
        #         {'params': model.base.parameters()},
        #         {'params': model.classifier.parameters(), 'lr': 1e-3}
        #     ], lr=1e-2, momentum=0.9)
        # scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
        scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                      factor=0.8, patience=10, min_lr=1e-8)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()

        # post-predictive regularizer
        # winsize = 5                 # each image being 5/2.5 seconds => 12.5 seconds
        # stepsize = 1                # step at an image per time
        # self.post_regularizer = Postalarm(winsize, stepsize, samplerate)
        # winsize_persamp = 5000
        # stepsize_persamp = 2500
        # numsamples = len(self.test_loader)
        # self.post_regularizer.compute_timepoints(winsize_persamp, stepsize_persamp, numsamples)

        # set lr scheduler, optimizer and loss function
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.criterion = criterion

        # if grad_clip:
        #     nn.utils.clip_grad_norm(self.net.parameters(), grad_clip)
        #     for p in self.net.parameters():
        #         p.data.add_(-self.learning_rate, p.grad.data)
        # log model
        self._log_model_tboard()

    def train(self, num_steps):
        """
        Main training function for pytorch
        """
        # summary for current training loop and a running average object for
        # loss
        summ = []
        loss_avg = utils.RunningAverage()
        # set model to training mode
        self.net.train()
        '''
        RUN TRAIN LOOP
        '''
        t = trange(num_steps)
        # t = range(num_steps)

        for step in t:
            # next(self.train_loader)
            (images, labels) = iter(self.train_loader).next()
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass -> get outputs and loss and get the loss
            outputs, _ = self.net(images)

            labels = labels.long()
            labels = torch.max(labels, 1)[1]
            # print(labels.shape)
            # print(outputs.shape)

            loss = self.criterion(outputs, labels)
            # clear the optimizer's holding of gradients and compute backprop
            # grad
            self.optimizer.zero_grad()
            loss.backward()

            # step the optimizer and scheduler
            self.optimizer.step()

            # Evaluate summaries only once in a while
            if step % self.save_summary_steps == 0:
                summ.append(
                    self._summarize(
                        outputs,
                        labels,
                        loss,
                        regularize=False))

            # update the average loss
            loss_avg.update(loss.data.item())
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))

        # compute mean of all metrics in summary
        metrics_mean = {metric: np.mean(
            [x[metric] for x in summ]) for metric in summ[0]}
        metrics_string = " ; ".join(
            "{}: {:05.3f}".format(
                k, v) for k, v in metrics_mean.items())
        self.logger.info("- Train metrics: " + metrics_string)
        return metrics_mean, images, loss

    def evaluate(self, num_steps):
        """Evaluate the model on `num_steps` batches.
        Args:
            num_steps: (int) number of batches to train on, each of size params.batch_size
        """
        # set model to evaluation mode
        self.net.eval()

        # summary for current eval loop
        summ = []

        # compute metrics over the dataset
        for i in range(num_steps):
            # fetch the next evaluation batch
            data_batch, labels_batch = iter(
                self.test_loader).next()  # next(self.test_loader)
            data_batch = data_batch.to(self.device)
            labels_batch = labels_batch.to(self.device)

            # compute model output
            output_batch, _ = self.net(data_batch)

            labels_batch = torch.max(labels_batch, 1)[1]
            labels_batch = labels_batch.long()

            # output_batch = output_batch.long()
            loss = self.criterion(output_batch, labels_batch)

            summ.append(
                self._summarize(
                    output_batch,
                    labels_batch,
                    loss,
                    regularize=True))

        # compute mean of all metrics in summary
        metrics_mean = {}
        for metric in summ[0]:
            metrics_mean[metric] = np.mean([x[metric] for x in summ])

        metrics_string = " ; ".join(
            "{}: {:05.3f}".format(
                k, v) for k, v in metrics_mean.items())
        self.logger.info("- Eval metrics : " + metrics_string)
        return metrics_mean, loss

    def train_and_evaluate(self, restore_file=None):
        # reload weights from restore_file if specified
        if restore_file is not None:
            restore_path = os.path.join(
                args.model_dir, args.restore_file + '.pth.tar')
            self.logger.info(
                "Restoring parameters from {}".format(restore_path))
            utils.load_checkpoint(restore_path, model, optimizer)

        best_val_acc = 0.0

        # tensorboard the initial convolutional layers
        # self._tboard_features(images, label, epoch, name='default')

        # run model through epochs / passes of the data
        for epoch in range(self.num_epochs):
            # Run one epoch
            self.logger.info("Epoch {}/{}".format(epoch + 1, self.num_epochs))

            ######################## 1. pass thru training ####################
            # compute number of batches in one epoch (one full pass over the
            # training set)
            num_steps_epoch = (self.train_size + 1) // self.batch_size
            self.logger.info(
                "Running training for {} steps".format(num_steps_epoch))
            train_metrics, images, train_loss = self.train(num_steps_epoch)

            self.logger.info('Epoch [{}/{}], Loss: {:.4f}'
                             .format(epoch + 1, self.num_epochs, train_loss.item()))
            self.logger.info('Acc: {:.2f}, Prec: {:.2f}, Recall: {:.2f}, FPR: {:.2f}'
                             .format(train_metrics['accuracy'], train_metrics['precision'], train_metrics['recall'], train_metrics['fp']))

            ######################## 2. pass thru validation ##################
            # Evaluate for one epoch on validation set
            num_steps = (self.val_size + 1) // self.batch_size
            self.logger.info(
                "Running validation for {} steps".format(num_steps))
            val_metrics, val_loss = self.evaluate(num_steps=num_steps)

            # determine if we should reduce lr based on validation
            self.scheduler.step(val_metrics['accuracy'])

            # get the metric we want to track over epochs
            val_acc = val_metrics['accuracy']
            is_best = val_acc >= best_val_acc
            self.logger.info('Epoch [{}/{}], Loss: {:.4f}'
                             .format(epoch + 1, self.num_epochs, val_loss.item()))
            self.logger.info('Acc: {:.2f}, Prec: {:.2f}, Recall: {:.2f}, FPR: {:.2f}'
                             .format(val_metrics['accuracy'], val_metrics['precision'], val_metrics['recall'], val_metrics['fp']))

            ######################## 3. Run post processing, checkpoints ######
            # Save weights
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'state_dict': self.net.state_dict(),
                                   'optim_dict': self.optimizer.state_dict()},
                                  is_best=is_best,
                                  checkpointdir=self.explogdir)

            # If best_eval, best_save_path
            if is_best:
                self.logger.info("- Found new best accuracy")
                best_val_acc = val_acc

                # Save best val metrics in a json file in the model directory
                best_json_path = os.path.join(
                    self.explogdir, "metrics_val_best_weights.json")
                utils.save_dict_to_json(val_metrics, best_json_path)

            # Save latest val metrics in a json file in the model directory
            last_json_path = os.path.join(
                self.explogdir, "metrics_val_last_weights.json")
            utils.save_dict_to_json(val_metrics, last_json_path)

            ######################## 4. TENSORBOARD LOGGING ###################
            # TENSORBOARD: loss, accuracy, precision, recall, fpr, values and
            # gradients
            self._tboard_metrics(
                train_loss,
                train_metrics,
                epoch,
                mode=constants.TRAIN)
            self._tboard_metrics(
                val_loss,
                val_metrics,
                epoch,
                mode=constants.VALIDATE)
            self._tboard_grad(epoch)

            # log output and the input everyevery <step> epochs
            if (epoch + 1) % self.save_summary_steps == 0:
                self._tboard_input(images, epoch)

        # tensorboard the convolutional layers after training
        # self._tboard_features(images, label, epoch, name='default')
        self.logger.info("Finished training!")

    

if __name__ == '__main__':
    from dnn_pytorch.models.nets.cnn import ConvNet
    from torchsummary import summary
    num_classes = 2
    imsize = 32
    n_colors = 4
    if device is None:
        # Device configuration
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # MOVE THE MODEL ONTO THE DEVICE WE WANT
    model = ConvNet(num_classes).to(device)

    num_epochs = 1
    batch_size = 32
    trainer = Trainer(model, num_epochs, batch_size, device=device)
    trainer.config()
    # Train the model
    trainer.train()

    # Test the model
    # trainer.test()
    # resultfilename = '{}_endmodel.ckpt'.format(patient)
    # trainer.save(resultfile)
