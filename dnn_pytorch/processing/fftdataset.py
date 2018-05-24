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

class FFT2DImageDataset(Dataset):
    '''
    Uses pytorch abstract class for representing our FFT image dataset.

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
        self.chanmeans = None
        self.chanstd = None

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

        self._loadalldata()

        if transform is not None:
            self.transform = transform
        else:
            transforms_to_use = [
                transforms.ToPILImage()#mode='RGBA'),
                augmentations.RandomLightingNoise(),
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
                transforms.ToTensor()
            ]
            if self.chanmeans is not None and self.chanstd is not None:
                transforms_to_use.append(transforms.Normalize(mean=self.chanmeans,    # apply normalization along channel axis
                                                             std=self.chanstd))
            transforms_to_use.append(augmentations.InjectNoise())

            data_transform = transforms.Compose(transforms_to_use)
            self.transform = data_transform

    def __len__(self):
        # return len(self.datasetnames)
        return len(self.datafiles)

        return len(self.X_train)

    def __getitem__(self, idx):
        sample = self.X_train[idx,...]
        ylabels = self.y_train[idx,...]

        # apply transformations to the dataset
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
        # so that the image_tensors channel x h x w
        image_tensors = image_tensors.swapaxes(1, 3)
        # lower sample by casting to 32 bits
        image_tensors = image_tensors.astype("float32")
        self.X_train = image_tensors
        self.y_train = ylabels
        self.class_weight = class_weight

        # call function to obtain avg/std along channel axis
        self._getchanstats()

    def _getchanstats(self):
        assert self.X_train.shape[2] == self.X_train.shape[3]
        chanaxis = 1
        numchans = self.X_train.shape[chanaxis]

        chanmeans = []
        chanstd = []
        for ichan in range(numchans):
            chandata = self.X_train[:, ichan, ...].ravel()
            chanmeans.append(np.mean(chandata))
            chanstd.append(np.std(chandata))

        self.chanmeans = chanmeans
        self.chanstd = chanstd

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