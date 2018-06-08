import torch
import torch.nn as nn
import sys
sys.path.append('../../../')
from dnn_pytorch.util.layer import Flatten, PrintLayer
from dnn_pytorch.base.constants.config import Config
from dnn_pytorch.base.utils.log_error import initialize_logger

class ConvNet(nn.Module):
    imsize = None
    n_colors = None
    num_classes = None

    def __init__(self, num_classes, imsize, n_colors, config=None):
        super(ConvNet, self).__init__()

        self.config = config or Config()
        self.logger = initialize_logger(
            self.__class__.__name__,
            self.config.out.FOLDER_LOGS)

        self.num_classes = num_classes
        self.n_colors = n_colors
        self.imsize = imsize

        # initialize a dropout layer
        self.dropout = nn.Dropout(p=0.5)

    def buildcnn(self):
        # VGG params
        n_layers = (4, 2, 1)
        self.pool_size = 2
        self.pool_stride = 2
        self.n_filters = 32
        self.kernel_size = 3
        cnn = self._buildvgg(n_layers)
        return cnn

    def buildoutput(self):
        n_size = self._get_conv_output(
            (self.n_colors, self.imsize, self.imsize))
        # create the output linear classification layers
        self.out = nn.Sequential()
        fc = nn.Linear(n_size, 512)
        self.out.add_module('fc', fc)
        self.out.add_module('dropout1', self.dropout)
        fc1 = nn.Linear(512, 256)
        self.out.add_module('fc1', fc1)
        self.out.add_module('dropout2', self.dropout)
        fc2 = nn.Linear(256, self.num_classes)
        self.out.add_module('fc2', fc2)
        self.out.add_module('dropout3', self.dropout)

    # generate input sample and forward to get shape
    def _get_conv_output(self, shape):
        bs = 1 # example batch size
        input = torch.autograd.Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _build_convolutional_block(self, idx, ilay, indim, outdim):
        conv = nn.Conv2d(in_channels=indim,                         # input height
                         out_channels=outdim,
                         kernel_size=self.kernel_size,                # filter size
                         stride=1, padding=2)                    # filter step, padding
        # apply Glorot/xavier uniform init
        torch.nn.init.xavier_uniform_(conv.weight)
        self.net.add_module('conv{}_{}'.format(idx, ilay), conv)
        self.net.add_module(
            'norm{}_{}'.format(
                idx, ilay), nn.BatchNorm2d(
                num_features=outdim))   # apply batch normalization
        # apply relu activation
        self.net.add_module(
            'activate{}_{}'.format(
                idx, ilay), nn.ReLU())

    def _buildvgg(self, n_layers=(4, 2, 1)):
        '''
        Model function for building up the VGG style CNN.
        
        if idx == 0 and ilay == 0:
            indim = self.n_colors
            self.dilation_size = 1
            self.padding = 1 #kernel_size-1
        else:
            indim = self.prevfilter
            self.dilation_size = self.dilation_size*2
            self.padding = self.padding*2
        '''
        self.net = nn.Sequential()

        # create the vgg-style convolutional layers w/ batch norm followed with
        # max-pooling
        for idx, n_layer in enumerate(n_layers):
            for ilay in range(n_layer):
                if idx == 0 and ilay == 0:
                    indim = self.n_colors
                else:
                    indim = outdim
                outdim = self.n_filters * (2**idx)        # n_filters to use
                # add convolutional blocks
                self._build_convolutional_block(idx, ilay, indim, outdim)

            self.net.add_module(
                'pool{}'.format(idx),
                nn.MaxPool2d(
                    kernel_size=self.pool_size,
                    stride=self.pool_stride))  # choose max value in poolsize area
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
    imsize = 64
    n_colors = 4
    expected_image_shape = (n_colors, imsize, imsize)
    input_tensor = torch.autograd.Variable(
        torch.rand(1, *expected_image_shape))

    cnn = ConvNet(num_classes=num_classes, imsize=imsize, n_colors=n_colors)
    cnn.buildcnn()
    cnn.buildoutput()
    # SUMMARIZE WEIGHTS IN EACH LAYER
    # for i, weights in enumerate(list(cnn.parameters())):
    #     print('i:',i,'weights:',weights.size())

    # PRINT FINAL OUTPUT USING A VARIABLE RUN THROUGH THE NETWORK
    # this call will invoke all registered forward hooks
    output_tensor, x = cnn(input_tensor)
    print("X shape: ", x.shape)
    # print(output_tensor.shape)
    print(cnn)

    # SUMMARIZE NETWORK USING KERAS STYLE SUMMARY
    summary(cnn, expected_image_shape)
