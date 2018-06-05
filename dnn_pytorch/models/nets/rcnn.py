import torch
import torch.nn as nn

from dnn_pytorch.util.layer import Flatten, PrintLayer
from dnn_pytorch.models.nets.cnn import ConvNet

class RecConvNet(nn.Module, ConvNet):
    cnns = None
    rnn = None
    output = None

    def __init__(self, num_classes, imsize, n_colors,
                 rnn_type, numwins=500, config=None):
        # super(RecConvNet, self).__init__()
        # self.config = config or Config()
        # self.logger = initialize_logger(self.__class__.__name__, self.config.out.FOLDER_LOGS)

        nn.Module.__init__()
        ConvNet.__init__(num_classes, imsize, n_colors, config=config)

        # RNN-LSTM params
        self.num_hidden_units = 1024
        self.numwins = numwins
        self.nlayers = 2
        self.nonlinearity = 'relu'

        # build sequence of convolutional networks
        cnns = []
        for i in range(numwins):
            cnns.append(self.buildcnn())
        self.cnns = cnns

        # build rnn
        self.buildrnn()

        # build output MLP
        self.buildoutput()

    def buildrnn(self):
        n_size = self._get_conv_output(
            (self.n_colors, self.imsize, self.imsize))
        self.rnn = nn.LSTM(n_size, self.num_hidden_units,
                           self.nlayers,
                           nonlinearity=self.nonlinearity,
                           dropout=0.5,
                           bidirectional=False)

    def preload_weights(self, state_dict):
        own_state = self.state_dict()
        for idx, cnn in enumerate(self.cnns):
            model_dict = cnn.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k,
                               v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            cnn.load_state_dict(pretrained_dict)

        self.logger.debug("Finished preloading state_dict for all cnns!")

     def buildoutput(self):
        n_size = self._get_conv_output(
            (self.n_colors, self.imsize, self.imsize))
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
        bs = 1
        input = torch.autograd.Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        """
        Helper function to get the output of one CNN
        """
        x = self.cnns[0](x)
        x = x.view(x.size(0), -1)  # to get the intermediate output
        return x

    def forward(self, x):
        x = self.cnns(x)
        x = x.view(x.size(0), -1)  # to get the intermediate output
        output = self.out(x)
        return output, x

if __name__ == '__main__':
    from torchsummary import summary
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    rcnn = RecConvNet()

    # SUMMARIZE WEIGHTS IN EACH LAYER
    # for i, weights in enumerate(list(cnn.parameters())):
    #     print('i:',i,'weights:',weights.size())

    # PRINT FINAL OUTPUT USING A VARIABLE RUN THROUGH THE NETWORK
    expected_image_shape = (4, 28, 28)
    input_tensor = torch.autograd.Variable(
        torch.rand(1, *expected_image_shape))
    # this call will invoke all registered forward hooks
    output_tensor, x = rcnn(input_tensor)
    print(x.shape)
    print(output_tensor.shape)
    print(cnn)

    # SUMMARIZE NETWORK USING KERAS STYLE SUMMARY
    summary(cnn, (4, 28, 28))
