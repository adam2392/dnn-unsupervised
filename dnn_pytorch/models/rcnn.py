import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, utils
from torchsummary import summary

from dnn_pytorch.util.layer import Flatten, PrintLayer
from dnn_pytorch.base.constants.config import Config
from dnn_pytorch.base.utils.log_error import initialize_logger

class RecConvNet(nn.Module):
    def __init__(self, num_classes=2, numwins=500, config=None):
        super(ConvNet, self).__init__()

        self.config = config or Config()
        self.logger = initialize_logger(self.__class__.__name__, self.config.out.FOLDER_LOGS)

        # RNN-LSTM params
        num_hidden_units = 1024
        self.numwins = numwins

        # CNN-VGG params
        w_init = True
        n_layers = (4,2,1)
        poolsize = 2
        n_filters = 32
        filter_size = 3
        self.num_classes = num_classes
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
                else:
                    indim = prevfilter
                prevfilter=n_filters*(2**idx) # keep track of the previous size filter
                conv = nn.Conv2d(in_channels=indim,                         # input height
                                out_channels=n_filters*(2**idx),        # n_filters to use
                                kernel_size=filter_size,                # filter size
                                stride=1, padding=2)                    # filter step, padding
                torch.nn.init.xavier_uniform_(conv.weight)              # apply Glorot/xavier uniform init
                self.net.add_module('conv{}_{}'.format(idx,ilay), conv)                                     
                self.net.add_module('norm{}_{}'.format(idx,ilay), nn.BatchNorm2d(num_features=prevfilter))   # apply batch normalization
                self.net.add_module('activate{}_{}'.format(idx,ilay), nn.ReLU())                                    # apply relu activation
            self.net.add_module('pool{}'.format(idx), nn.MaxPool2d(kernel_size=poolsize, stride=poolsize)) # choose max value in poolsize area

        # create the output linear classification layers
        self.out = nn.Sequential()
        fc = nn.Linear(128*6*6, 512)
        self.out.add_module('fc',fc)
        self.out.add_module('dropout', nn.Dropout(p=0.5))
        fc2 = nn.Linear(512, self.num_classes)
        self.out.add_module('fc2',fc2)    
        self.out.add_module('dropout',nn.Dropout(p=0.5))

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1) # to get the intermediate output
        output = self.out(x)
        return output, x

class Train(object):
    def __init__(self, net, device=None, config=None):
        self.config = config or Config()
        self.logger = initialize_logger(self.__class__.__name__, self.config.out.FOLDER_LOGS)

        if device is None:
            # Device configuration
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.net = net
        # Hyper parameters
        self.num_epochs = 100
        self.batch_size = 64
        self.learning_rate = 1e-4

    def composedatasets(self, train_dataset_obj, test_dataset_obj):
        # dataset = FFT2DImageDataset(root_dir, datasetnames, transform=data_transform)
        # MNIST dataset
        # self.train_dataset = torchvision.datasets.MNIST(root='../../data/',
        #                                            train=True, 
        #                                            transform=transforms.ToTensor(),
        #                                            download=True)
        # self.test_dataset = torchvision.datasets.MNIST(root='../../data/',
        #                                           train=False, 
        #                                           transform=transforms.ToTensor())

        self.train_loader = DataLoader(train_dataset_obj, 
                            batch_size=self.batch_size,
                            shuffle=True, 
                            num_workers=1)
        self.test_loader = DataLoader(test_dataset_obj, 
                            batch_size=self.batch_size,
                            shuffle=True, 
                            num_workers=1)

    def train(self):
        optimparams = {
            'lr': learning_rate
        }
        optimizer = torch.optim.Adam(self.net.parameters(), 
                                    **optimparams)
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()

        # Train the model
        total_step = len(self.train_loader)

        # run model through epochs / passes of the data
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass -> get outputs and loss
                outputs = self.net(images)
                loss = criterion(outputs, labels)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # log output every <step> epochs
                if (i+1) % 10 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                           .format(epoch+1, self.num_epochs, i+1, total_step, loss.item()))
                    # print('Epoch [{}/{}], Step [{}/{}], Gradients: {:.4f}' 
                    #        .format(epoch+1, self.num_epochs, i+1, total_step,  ))

    def test(self):
        testlen = len(self.test_loader)

        # Test the model
        self.net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Test Accuracy of the model on the {} test images: {} %'.format(testlen, 100 * correct / total))

        # Save the model checkpoint
        torch.save(self.net.state_dict(), 'model.ckpt')

if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    rcnn = RecConvNet()

    ## SUMMARIZE WEIGHTS IN EACH LAYER
    # for i, weights in enumerate(list(cnn.parameters())):
    #     print('i:',i,'weights:',weights.size())

    ## PRINT FINAL OUTPUT USING A VARIABLE RUN THROUGH THE NETWORK
    expected_image_shape = (4, 28, 28)
    input_tensor = torch.autograd.Variable(torch.rand(1, *expected_image_shape))
    # this call will invoke all registered forward hooks
    output_tensor, x = rcnn(input_tensor)
    print(x.shape)
    print(output_tensor.shape)
    print(cnn)

    ## SUMMARIZE NETWORK USING KERAS STYLE SUMMARY
    summary(cnn, (4, 28, 28))


