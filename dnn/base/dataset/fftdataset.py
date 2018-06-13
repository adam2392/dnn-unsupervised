import os
import sklearn
import numpy as np

# Ignore warnings
import warnings
from dnn.util import augmentations
from dnn.base.constants.config import Config
from dnn.base.utils.log_error import initialize_logger
import dnn.base.constants.model_constants as constants

class FFT2DImageDataset(Dataset):
    '''
    Uses pytorch abstract class for representing our FFT image dataset.

    '''
    chanmeans = None
    chanstd = None
    imsize = None
    n_colors = None

    X_train = None
    X_test = None
    y_train = None
    y_test = None

    def __init__(self, data_tensor, ylabels, config=None):
        """
        Args:
            root_dir (string): directory with all the data
            datasetnames (list of strings): list of the datasets we want to use
            transform (callable, optional): Optional transform to be applied
            on a sample
        """
        self.config = config or Config()
        self.logger = initialize_logger(self.__class__.__name__, self.config.out.FOLDER_LOGS)

        assert len(data_tensor) == len(ylabels)
        self.data_tensor = data_tensor
        self.ylabels = ylabels
        # call function to obtain avg/std along channel axis
        self._getchanstats()

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, idx):
        sample = self.data_tensor[idx,...]
        ylabels = self.ylabels[idx,...]
        return sample, ylabels

    @property
    def imsize(self):
        return self.data_tensor.shape[2]

    @property
    def n_colors(self):
        return self.data_tensor.shape[3]

    def _getchanstats(self):
        assert self.data_tensor.shape[2] == self.data_tensor.shape[1]
        chanaxis = 1
        numchans = self.data_tensor.shape[chanaxis]

        chanmeans = []
        chanstd = []
        for ichan in range(numchans):
            chandata = self.data_tensor[...,ichan].ravel()
            chanmeans.append(np.mean(chandata))
            chanstd.append(np.std(chandata))
        self.chanmeans = np.array(chanmeans)
        self.chanstd = np.array(chanstd)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from util import augmentations
    import torchvision
    from torchvision import transforms, utils
    

    root_dir = '/Volumes/ADAM LI/pydata/'
    datasetnames = []

    chanmeans = [0.485, 0.456, 0.406]
    chanstd = [0.229, 0.224, 0.225]
    imsize = 28

    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        #     transforms.RandomApply(transforms, p=0.5),
    #     augmentations.RandomLightingNoise(),
        transforms.RandomSizedCrop(2),  
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.CenterCrop(3),
        transforms.RandomRotation(degrees=5, 
                                resample=False, 
                                expand=False, 
                                center=None),
        transforms.RandomAffine(degrees=5, 
                                translate=(0.1,0.1), 
                                scale=None, 
                                shear=5, 
                                resample=False, 
                                fillcolor=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=chanmeans,    # apply normalization along channel axis
                             std=chanstd),
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])


    dataset = FFT2DImageDataset(root_dir, datasetnames, transform=None)
    dataloader = DataLoader(dataset, 
                        batch_size=4,
                        shuffle=True, 
                        num_workers=3)

    # Helper function to show a batch
    def show_landmarks_batch(sample_batched):
        """Show image with landmarks for a batch of samples."""
        images_batch, ylabels = sample_batched[0], sample_batched[1]
        
        # get the batch size and imsize
        batch_size = len(images_batch)
        im_size = images_batch.size(2)
        print("Batch size and im size are: %s %s" % (batch_size, im_size))

        images_batch = images_batch[0,0,...]
        images_batch = np.swapaxes(images_batch, 0,2)
        images_batch = images_batch[:,np.newaxis,...]
        images_batch = [im for im in images_batch]
        grid = utils.make_grid(images_batch, normalize=True)
        print(grid.shape)
        img = grid.numpy().transpose((1, 2, 0))
        print(img.shape)
        plt.imshow(img, cmap='jet')
        plt.title('Batch from dataloader')

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched[0].size(), sample_batched[1].size())

        # observe 0th batch and stop.
        if i_batch == 0:
            plt.figure()
            show_landmarks_batch(sample_batched)
            plt.axis('off')
    #         plt.colorbar()
    #         plt.ioff()
            plt.show()
            break