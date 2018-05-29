import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, StepLR, ReduceLROnPlateau

from dnn_pytorch.models.trainers.base import BaseTrainer
from dnn_pytorch.util import utils
from dnn_pytorch.models.evaluate import evaluate
from dnn_pytorch.models.metrics.classifier import BinaryClassifierMetric
import dnn_pytorch.base.constants.model_constants as constants

# import tensorboard for writing stuff
from tensorboardX import SummaryWriter
from tqdm import trange

class Trainer(BaseTrainer):
    grad_queue = []
    metric_comp = BinaryClassifierMetric()
    
    # create a dictionary of metrics with their corresponding "lambda" functions
    metrics = {
        'accuracy': metric_comp._accuracy,
        'recall': metric_comp._recall,
        'precision': metric_comp._precision,
        'fp': metric_comp._fp
    }
    # create objects to hold our train/test metrics
    trainmetrics = TrainMetrics()
    testmetrics = TestMetrics()

    def __init__(self, net, num_epochs, batch_size, 
                device=None, 
                tboard_log_name=None, comment='',
                explogdir=None,
                learning_rate=constants.LEARNING_RATE,
                dropout=constants.DROPOUT,
                shuffle=constants.SHUFFLE,
                config=None):
        super(Trainer, self).__init__(net=net,
                                      device=device,
                                      config=config)

        # Hyper parameters - training
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # hyper parameters - dataset
        self.shuffle = shuffle 

        # set tensorboard writer
        tboard_log_name = 'tboard_logs'
        self.writer = SummaryWriter(os.path.join(self.config.tboard.FOLDER_LOGS, tboard_log_name),
                                    comment=comment)

        # use tensorflow bboard logger
        self.explogdir = explogdir
        self._log_model_tboard()

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
        self.imsize = self.train_loader.imsize
        self.n_colors = self.train_loader.n_colors

        self.logger.info("Setting the datasets for training/testing in trainer object!")
        self.logger.info("Image size is {} with {} colors".format(imsize, n_colors))

    def loadmetrics(self, y_true, y_pred, metricholder):
        self.metric_comp.compute_scores(y_true, y_pred)

        # add to list for the metrics
        self.metricholder.recall_queue.append(self.metrics.recall)
        self.metricholder.precision_queue.append(self.metrics.precision)
        self.metricholder.fp_queue.append(self.metrics.fp)
        self.metricholder.accuracy_queue.append(self.metrics.accuracy)

    def config(self):
        optimparams = {
            'lr': learning_rate,
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
                                    factor=0.8, patient=10, min_lr=1e-8)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()

        # set lr scheduler, optimizer and loss function
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.criterion = criterion

    def train(self, num_steps):
        """
        Main training function for pytorch
        """
        # summary for current training loop and a running average object for loss
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
            (images, labels) = next(self.train_loader)
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass -> get outputs and loss and get the loss
            outputs = self.net(images)
            loss = self.criterion(outputs, labels)
            # clear the optimizer's holding of gradients and compute backprop grad
            self.optimizer.zero_grad()
            loss.backward()

            # step the optimizer and scheduler
            self.optimizer.step()
            self.scheduler.step()

            # Evaluate summaries only once in a while
            if step % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = outputs.data.cpu().numpy()
                labels_batch = labels.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric:self.metrics[metric](output_batch, labels_batch)
                                 for metric in self.metrics}
                summary_batch['loss'] = loss.data[0]
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.data[0])
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))

        # compute mean of all metrics in summary
        metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
        self.logging.info("- Train metrics: " + metrics_string)
        return metrics_mean

    def evaluate(self, num_steps):
        """Evaluate the model on `num_steps` batches.
        Args:
            model: (torch.nn.Module) the neural network
            loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
            data_iterator: (generator) a generator that generates batches of data and labels
            metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
            params: (Params) hyperparameters
            num_steps: (int) number of batches to train on, each of size params.batch_size
        """

        # set model to evaluation mode
        self.net.eval()

        # summary for current eval loop
        summ = []

        # compute metrics over the dataset
        for _ in range(num_steps):
            # fetch the next evaluation batch
            data_batch, labels_batch = next(self.test_loader)
            data_batch = data_batch.to(self.device)
            labels_batch = labels_batch.to(self.device)
            
            # compute model output
            output_batch = self.net(data_batch)
            loss = self.criterion(output_batch, labels_batch)

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            # compute all metrics on this batch
            summary_batch = {metric: self.metrics[metric](output_batch, labels_batch)
                             for metric in self.metrics}
            summary_batch['loss'] = loss.data[0]
            summ.append(summary_batch)

        # compute mean of all metrics in summary
        metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
        self.logging.info("- Eval metrics : " + metrics_string)
        return metrics_mean

    def train_and_evaluate(self, restore_file=None):
        # reload weights from restore_file if specified
        if restore_file is not None:
            restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
            self.logging.info("Restoring parameters from {}".format(restore_path))
            utils.load_checkpoint(restore_path, model, optimizer)

        best_val_acc = 0.0

        # size of training/testing set
        train_size = len(self.train_loader)
        val_size = len(self.test_loader)

        # tensorboard the initial convolutional layers
        # self._tboard_features(images, label, epoch, name='default')

        # run model through epochs / passes of the data
        for epoch in range(self.num_epochs):
            # Run one epoch
            self.logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

            ######################## 1. pass thru training ########################
            # compute number of batches in one epoch (one full pass over the training set)
            num_steps_epoch = (train_size + 1) // self.batch_size
            train_metrics = self.train(num_steps_epoch)

            self.logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, self.num_epochs, step+1, total_step, loss.item()))
            self.logger.info('Acc: {:.2f}, Prec: {:.2f}, Recall: {:.2f}, FPR: {:.2f}' 
                   .format(train_metrics['accuracy'], train_metrics['precision'], train_metrics['recall'], train_metrics['fp']))

            ######################## 2. pass thru validation ########################
             # Evaluate for one epoch on validation set
            num_steps = (val_size + 1) // self.batch_size
            val_metrics = self.evaluate(num_steps=num_steps)
            
            # get the metric we want to track over epochs
            val_acc = val_metrics['accuracy']
            is_best = val_acc >= best_val_acc
            self.logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, self.num_epochs, step+1, total_step, loss.item()))
            self.logger.info('Acc: {:.2f}, Prec: {:.2f}, Recall: {:.2f}, FPR: {:.2f}' 
                   .format(val_metrics['accuracy'], val_metrics['precision'], val_metrics['recall'], val_metrics['fp']))


            ######################## 3. Run post processing, checkpoints ########################
            # Save weights
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'state_dict': model.state_dict(),
                                   'optim_dict' : optimizer.state_dict()}, 
                                   is_best=is_best,
                                   checkpoint=self.explogdir)

            # If best_eval, best_save_path        
            if is_best:
                self.logging.info("- Found new best accuracy")
                best_val_acc = val_acc
                
                # Save best val metrics in a json file in the model directory
                best_json_path = os.path.join(self.explogdir, "metrics_val_best_weights.json")
                utils.save_dict_to_json(val_metrics, best_json_path)

            # Save latest val metrics in a json file in the model directory
            last_json_path = os.path.join(self.explogdir, "metrics_val_last_weights.json")
            utils.save_dict_to_json(val_metrics, last_json_path)

            ######################## 4. TENSORBOARD LOGGING ########################
            # TENSORBOARD: loss, accuracy, precision, recall, fpr, values and gradients
            self._tboard_metrics(loss, train_metrics, epoch, mode=constants.TRAIN)
            self._tboard_metrics(loss, val_metrics, epoch, mode=constants.VALIDATE)
            self._tboard_grad(epoch)

            # log output and the input everyevery <step> epochs
            if (epoch+1) % 10 == 0:
                self._tboard_input(images, epoch)
                
        # tensorboard the convolutional layers after training
        # self._tboard_features(images, label, epoch, name='default')
        self.logger.info("Finished training!")

    def save(self, resultfile):
        # Save the model checkpoint
        torch.save(self.net.state_dict(), 'model.ckpt')

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
    resultfile = os.path.join(resultdatadir, '{}_endmodel.ckpt'.format(expname))
    trainer.save(resultfile)
