import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                      shuffle=False)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)
class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    def forward(self, x):
        # Do your print / debug stuff here
        print(x)
        return x
class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()

        # VGG params
        w_init = True
        n_layers = (4,2,1)
        poolsize = 2
        n_filters = 32
        filter_size = 3
        self._buildvgg(w_init, n_layers, poolsize, n_filters, filter_size)

    def _buildvgg(self, w_init=True, n_layers=(4,2,1), poolsize=2, n_filters=32, filter_size=3):
        '''
        Model function for building up the VGG style CNN.

        To Do:
        - Consider switching code layout to: 

            layers = []
            layers.append(nn.Linear(3, 4))
            layers.append(nn.Sigmoid())
            layers.append(nn.Linear(4, 1))
            layers.append(nn.Sigmoid())

            net = nn.Sequential(*layers)

        '''
        self.net = nn.Sequential()

        # create the vgg-style convolutional layers w/ batch norm followed with max-pooling
        for idx, n_layer in enumerate(n_layers):
            for ilay in range(n_layer):
                if idx == 0 and ilay == 0:
                    indim = 4
                elif ilay == 0:
                    indim = n_filters*(2**(idx-1))
                else:
                    indim = n_filters*(2**idx)
                conv = nn.Conv2d(in_channels=indim,                         # input height
                                out_channels=n_filters*(2**idx),        # n_filters to use
                                kernel_size=filter_size,                # filter size
                                stride=1, padding=2)                    # filter step, padding
                torch.nn.init.xavier_uniform_(conv.weight)              # apply Glorot/xavier uniform init
                self.net.add_module('conv{}_{}'.format(idx,ilay), conv)                                     
                self.net.add_module('norm{}_{}'.format(idx,ilay), nn.BatchNorm2d(num_features=n_filters))   # apply batch normalization
                self.net.add_module('activate{}_{}'.format(idx,ilay), nn.ReLU())                                    # apply relu activation
            self.net.add_module('pool{}'.format(idx), nn.MaxPool2d(kernel_size=poolsize, stride=poolsize)) # choose max value in poolsize area

        self.net.add_module('print', PrintLayer())
        # create the fully-connected layers at the end
        # self.net.add_module(Flatten())
        # fc = nn.SoftMax()
        # self.net.add_module(fc)
        # self.net.add_module(nn.Dropout(p=0.5))
        # fc = nn.Linear(fc.size(), num_classes)
        # self.net.add_module(fc)
        # self.net.add_module(nn.Dropout(p=0.5))
        # for idx, module in self.net.named_children():
        #   print(self.net.module[idx].output.size(), module)

        # self.out = nn.Sequential()
        # # fc = nn.Softmax()
        # # self.out.add_module('softmax', fc)
        # # self.out.add_module('dropout', nn.Dropout(p=0.5))
        # fc = nn.Linear(fc.size(), num_classes)
        # self.out.add_module('fc',fc)
        # self.out.add_module('dropout',nn.Dropout(p=0.5))

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return x
        # output = self.out(x)
        # return output, x

if __name__ == '__main__':
    cnn = ConvNet()
    if torch.cuda.is_available():
        cnn.cuda()
    print(cnn)
    summary(cnn, (4, 28, 28))


