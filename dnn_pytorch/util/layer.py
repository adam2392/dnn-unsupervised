import torch.nn as nn

class Flatten(nn.Module):
    """
    Customized pytorch layer for flattening the output of previous layer
    """
    def forward(self, x):
        return x.view(x.size()[0], -1)

class PrintLayer(nn.Module):
    """
    Customized pytorch layer for printing the current computational graph
    """
    def __init__(self):
        super(PrintLayer, self).__init__()
    def forward(self, x):
        # Do your print / debug stuff here
        print(x)
        return x