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
        self.logger = initialize_logger(self.__class__.__name__, self.config.out.FOLDER_LOGS)
        
        if device is None:
            # Device configuration
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.logger.info("Setting the device to be {}".format(self.device))

        self.net = net
    # ================================================================== #
    #                        Tensorboard Logging                         #
    # ================================================================== #
    def _tboard_metrics(self, loss, metrics, step, mode=constants.TRAIN):
         # compute mean of all metrics in summary
        metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
        # 1. Log scalar values (scalar summary)
        info = {metric: metrics[metric] for metric in metrics.keys()}
        info['loss'] = loss.item()

        # log each item
        for tag, value in info.items():
            self.writer.add_scalar(tag+'/'+mode, value, step+1)

    def _tboard_grad(self, step):
        # 2. Log values and gradients of the parameters (histogram summary)
        for tag, value in self.net.named_parameters():
            tag = tag.replace('.', '/')
            self.writer.add_histogram(tag, value.data.cpu().numpy(), step+1)
            self.writer.add_histogram(tag+'/grad', value.grad.data.cpu().numpy(), step+1)

    def _tboard_input(self, images, step):
        # 3. Log training images (image summary)
        info = { 
                'images': images.view(-1, self.imsize, self.imsize)[:5].cpu().numpy() 
            }
        for tag, images in info.items():
            self.writer.add_image(tag, images, step+1)

    def _tboard_features(self, images, label, step, name='default'):
        # 4. add embedding:
        self.writer.add_embedding(images.ravel(), 
                                metadata=label, 
                                label_img=images.unsqueeze(1), 
                                global_step=step+1,
                                name=name)

    def _log_model_tboard(self):
        # log model to tensorboard
        expected_image_shape = (self.batch_size, self.n_colors, self.imsize, self.imsize)
        input_tensor = torch.autograd.Variable(torch.rand(*expected_image_shape),
                                    requires_grad=True)
        # this call will invoke all registered forward hooks
        # res = self.net(input_tensor)
        self.writer.add_graph(self.net, input_tensor)
