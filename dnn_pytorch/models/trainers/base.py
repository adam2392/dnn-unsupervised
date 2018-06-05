import torch
from dnn_pytorch.base.constants.config import Config, OutputConfig
from dnn_pytorch.base.utils.log_error import initialize_logger
import dnn_pytorch.base.constants.model_constants as constants


class TrainMetrics(object):
    # metrics
    loss_queue = []
    recall_queue = []
    precision_queue = []
    accuracy_queue = []
    fp_queue = []


class TestMetrics(object):
    # metrics
    loss_queue = []
    recall_queue = []
    precision_queue = []
    accuracy_queue = []
    fp_queue = []


class BaseTrainer(object):
    def __init__(self, net, device=None, config=None):
        self.config = config or Config()
        self.logger = initialize_logger(
            self.__class__.__name__,
            self.config.out.FOLDER_LOGS)

        if device is None:
            # Device configuration
            device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.logger.info("Setting the device to be {}".format(self.device))

        self.net = net

    def _summarize(self, outputs, labels, loss, regularize=False):
        # extract data from torch Variable, move to cpu, convert to numpy
        # arrays
        output_batch = outputs.data.cpu().numpy()
        labels_batch = labels.data.cpu().numpy()

        # ensure that metrics only take in the predicted labels
        # labels_batch = np.argmax(labels_batch,1)
        output_batch = np.argmax(output_batch, 1)

        # regularize the output
        if self.post_regularizer is not None and regularize is True:
            self.post_regularizer.load_predictions(output_batch)
            alarm_inds = self.post_regularizer.temporal_smooth(thresh=0.5)
            output_batch[:] = 0
            output_batch[alarm_inds] = 1

        # compute all metrics on this batch
        summary_batch = {metric: self.metrics[metric](output_batch, labels_batch)
                         for metric in self.metrics}
        summary_batch['loss'] = loss.data.item()
        return summary_batch
    # ================================================================== #
    #                        Tensorboard Logging                         #
    # ================================================================== #

    def _tboard_metrics(self, loss, metrics, step,
                        mode=constants.TRAIN, on=True):
        if on:
            # 1. Log scalar values (scalar summary)
            info = {metric: metrics[metric] for metric in metrics.keys()}
            info['loss'] = loss.item()

            # log each item
            for tag, value in info.items():
                self.writer.add_scalar(tag + '/' + mode, value, step + 1)

    def _tboard_grad(self, step, on=True):
        if on:
            # 2. Log values and gradients of the parameters (histogram summary)
            for tag, value in self.net.named_parameters():
                tag = tag.replace('.', '/')
                self.writer.add_histogram(
                    tag, value.data.cpu().numpy(), step + 1)
                self.writer.add_histogram(
                    tag + '/grad', value.grad.data.cpu().numpy(), step + 1)

    def _tboard_input(self, images, step, on=True):
        if on:
            # 3. Log training images (image summary)
            info = {
                'images': images.view(-1, self.imsize, self.imsize)[:5].cpu().numpy()
            }
            for tag, images in info.items():
                self.writer.add_image(tag, images, step + 1)

    def _tboard_features(self, images, label, step, name='default', on=True):
        if on:
            # 4. add embedding:
            self.writer.add_embedding(images.ravel(),
                                      metadata=label,
                                      label_img=images.unsqueeze(1),
                                      global_step=step + 1,
                                      name=name)

    def _log_model_tboard(self, on=True):
        if on:
            # log model to tensorboard
            expected_image_shape = (
                self.batch_size,
                self.n_colors,
                self.imsize,
                self.imsize)
            input_tensor = torch.autograd.Variable(torch.rand(*expected_image_shape),
                                                   requires_grad=True)
            # this call will invoke all registered forward hooks
            # res = self.net(input_tensor)
            self.writer.add_graph(self.net, input_tensor)

    def save(self, resultfilename):
        if self.outputdatadir not in resultfilename:
            resultfilename = os.path.join(self.outputdatadir, resultfilename)
        # Save the model checkpoint
        torch.save(self.net.state_dict(), resultfilename)
