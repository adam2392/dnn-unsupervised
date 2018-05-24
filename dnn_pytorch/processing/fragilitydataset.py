import torch
from torch.utils.data import Dataset, DataLoader

import os
import sklearn
import numpy as np
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from dnn_pytorch.base.constants.config import Config
from dnn_pytorch.base.utils.log_error import initialize_logger

class FragilityImageDataset(Dataset):
    '''
    Uses pytorch abstract class for representing our Fragility image dataset.

    '''
    def __init__(self, root_dir, datasetnames=None, transform=None, config=None):
        """
        Args:
            root_dir (string): directory with all the data
            datasetnames (list of strings): list of the datasets we want to use
            transform (callable, optional): Optional transform to be applied
            on a sample
        """
        self.config = config or Config()
        self.logger = initialize_logger(self.__class__.__name__, self.config.out.FOLDER_LOGS)

        self.root_dir = root_dir
        self.datasetnames = datasetnames
        self.transform = transform

        # get all the datafiles avail
        datafiles = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if datasetnames is None:
                    if any(name in file for name in datasetnames):
                        datafiles.append(os.path.join(root, file))
                else:
                    datafiles.append(os.path.join(root, file))
        self.datafiles = datafiles

        # load all the data from directory to RAM
        self._loadalldata()

    def __len__(self):
        # return len(self.datasetnames)
        return len(self.datafiles)

    def __getitem__(self, idx):
        # datafile = self.datafiles[idx]
        # datastruct = np.load(datafile)
        # sample = datastruct['image_tensor']
        # metadata = datastruct['metadata'].item()
        
        # # get the necessary data
        # ylabels = metadata['ylabels']

        # # reshape the sample to make sure they work
        # sample = sample.swapaxes(1, 3)
        # # lower sample by casting to 32 bits
        # sample = sample.astype("float32")

        # # load the ylabeled data 1 in 0th position is 0, 1 in 1st position is 1
        # invert_y = 1 - ylabels
        # ylabels = np.concatenate((invert_y, ylabels), axis=1)

        # # print(datafile)
        # # print(sample.shape)
        # # print(ylabels.shape)
        # # apply transformation augmentation if set
        # if self.transform:
        #     sample = self.transform(sample)

        # return sample, ylabels
        sample = self.X_train[idx,...]
        ylabels = self.y_train[idx,...]

        # apply transformation
        if self.transform:
            sample = self.transform(sample)
        return sample, ylabels

    def _loadalldata(self):
        for idx, datafile in enumerate(self.datafiles):
            datastruct = np.load(datafile)
            image_tensor = datastruct['image_tensor']
            metadata = datastruct['metadata'].item()

            if idx == 0:
                image_tensors = image_tensor
                ylabels = metadata['ylabels']
            else:
                image_tensors = np.append(image_tensors, image_tensor, axis=0)
                ylabels = np.append(ylabels, metadata['ylabels'], axis=0)

            break

        # load the ylabeled data 1 in 0th position is 0, 1 in 1st position is 1
        invert_y = 1 - ylabels
        ylabels = np.concatenate((invert_y, ylabels), axis=1)
        # format the data correctly
        class_weight = sklearn.utils.compute_class_weight('balanced',
                                                          np.unique(
                                                              ylabels).astype(int),
                                                          np.argmax(ylabels, axis=1))
        image_tensors = image_tensors.swapaxes(1, 3)
        # lower sample by casting to 32 bits
        image_tensors = image_tensors.astype("float32")
        self.X_train = image_tensors
        self.y_train = ylabels
        self.class_weight = class_weight

if __name__ == '__main__':
    from skimage import io, transform
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
        transforms.ToPILImage(mode='RGBA'),
    #     transforms.RandomApply(transforms, p=0.5),
    #     augmentations.RandomLightingNoise(),
    #     transforms.RandomSizedCrop(2),  
    #     transforms.CenterCrop(3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
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
        augmentations.InjectNoise(),
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
