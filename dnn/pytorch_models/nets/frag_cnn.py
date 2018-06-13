import sys
sys.path.append('../../../')
import torch
import torch.nn as nn

from dnn_pytorch.util.layer import Flatten, PrintLayer
from dnn_pytorch.base.constants.config import Config
from dnn_pytorch.base.utils.log_error import initialize_logger

class Conv1D(nn.Module):
    seqsize = None
    n_colors = None
    num_classes = None

    dilation_size = 1
    padding = 0

    def __init__(self, num_classes, seqsize, n_colors, config=None):
        super(Conv1D, self).__init__()

        self.config = config or Config()
        self.logger = initialize_logger(
            self.__class__.__name__,
            self.config.out.FOLDER_LOGS)

        self.num_classes = num_classes
        self.n_colors = n_colors
        self.seqsize = seqsize

        # initialize a dropout layer
        self.dropout = nn.Dropout(p=0.5)

    def buildcnn(self):
        # VGG params
        n_layers = (8, )
        n_filters = 16
        filter_size = 3
        cnn = self._buildmodel(n_layers, n_filters, filter_size)
        return cnn

    def buildoutput(self):
        n_size = self._get_conv_output((self.n_colors, self.seqsize))
        # create the output linear classification layers
        self.out = nn.Sequential()
        fc = nn.Linear(n_size, 512)
        self.out.add_module('fc', fc)
        self.out.add_module('dropout', self.dropout)
        fc2 = nn.Linear(512, self.num_classes)
        self.out.add_module('fc2', fc2)
        self.out.add_module('dropout', self.dropout)

    # generate input sample and forward to get shape
    def _get_conv_output(self, shape):
        bs = 1 # example batch size
        input = torch.autograd.Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def __build_residual_block(self, idx, ilay, n_filters, filter_size):
        if idx == 0 and ilay == 0:
            indim = self.n_colors
        else:
            indim = self.prevfilter

        # keep track of the previous size filter
        self.prevfilter = n_filters * (2**idx)
        conv = nn.Conv1d(in_channels=indim,                         # input height
                         out_channels=n_filters*(2**idx),        # n_filters to use
                         kernel_size=filter_size,                # filter size
                         stride=1, 
                         padding=0)                    # filter step, padding
        # apply Glorot/xavier uniform init
        torch.nn.init.xavier_uniform_(conv.weight)
        self.net.add_module('conv{}_{}'.format(idx, ilay), conv)
        self.net.add_module(
            'norm{}_{}'.format(
                idx, ilay), nn.BatchNorm1d(
                num_features=self.prevfilter))   # apply batch normalization
        # apply relu activation
        self.net.add_module(
            'activate{}_{}'.format(
                idx, ilay), nn.ReLU())
        # apply dropout
        self.net.add_module('dropout{}_{}'.format(idx, ilay), self.dropout)

    def __build_residual_block_dilations(self, idx, ilay, n_filters, filter_size):
        if idx == 0 and ilay == 0:
            indim = self.n_colors
            self.dilation_size = 1
            self.padding = 1 #filter_size-1
        else:
            indim = self.prevfilter
            self.dilation_size = self.dilation_size*2
            self.padding = self.padding*2

        # keep track of the previous size filter
        self.prevfilter = n_filters * (2**idx)
        conv = nn.Conv1d(in_channels=indim,                         # input height
                         out_channels=n_filters*(2**idx),        # n_filters to use
                         kernel_size=filter_size,                # filter size
                         stride=1, dilation=self.dilation_size, 
                         padding=self.padding)                    # filter step, padding
        # apply Glorot/xavier uniform init
        torch.nn.init.xavier_uniform_(conv.weight)
        self.net.add_module('conv{}_{}'.format(idx, ilay), conv)
        self.net.add_module(
            'norm{}_{}'.format(
                idx, ilay), nn.BatchNorm1d(
                num_features=self.prevfilter))   # apply batch normalization
        # apply relu activation
        self.net.add_module(
            'activate{}_{}'.format(
                idx, ilay), nn.ReLU())
        # apply dropout
        self.net.add_module('dropout{}_{}'.format(idx, ilay), self.dropout)

    def _buildmodel(self, n_layers, n_filters, filter_size):
        '''
        Model function for building up the 1D CNN.

        '''
        self.net = nn.Sequential()

        # build residual blocks
        for idx, n_layer in enumerate(n_layers):
            for ilay in range(n_layer):
                self.__build_residual_block(idx, ilay, n_filters, filter_size)
        return self.net

    def _forward_features(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)  # to get the intermediate output
        return x

    def forward(self, x):
        output = None
        x = self._forward_features(x)
        output = self.out(x)
        return output, x

if __name__ == '__main__':
    from torchsummary import summary
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_classes = 2
    seqsize = 500
    n_colors = 4
    expected_image_shape = (n_colors, seqsize)
    input_tensor = torch.autograd.Variable(
        torch.rand(1, *expected_image_shape))
 
    cnn = Conv1D(num_classes, seqsize, n_colors)
    cnn.buildcnn()

    # x = cnn._forward_features(input_tensor)
    # print(x)
    cnn.buildoutput()

    # PRINT FINAL OUTPUT USING A VARIABLE RUN THROUGH THE NETWORK
    # this call will invoke all registered forward hooks
    output_tensor, x = cnn(input_tensor)
    print("X shape: ", x.shape)
    # print(output_tensor.shape)
    print(cnn)

    # SUMMARIZE NETWORK USING KERAS STYLE SUMMARY
    summary(cnn, (n_colors, seqsize))
