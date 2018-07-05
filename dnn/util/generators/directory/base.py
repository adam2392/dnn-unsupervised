import numpy as np 
import keras
from keras import backend as K
import multiprocessing
import os
import threading
import warnings
import multiprocessing.pool
from functools import partial
from dnn.util.generators.base.baseseq import Iterator, DirectoryIterator

from random import shuffle

def _iter_valid_files(directory, white_list_formats, follow_links):
    """Iterates on files with extension in `white_list_formats` contained in `directory`.

    # Arguments
        directory: Absolute path to the directory
            containing files to be counted
        white_list_formats: Set of strings containing allowed extensions for
            the files to be counted.
        follow_links: Boolean.

    # Yields
        Tuple of (root, filename) with extension in `white_list_formats`.
    """
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=follow_links),
                      key=lambda x: x[0])

    for root, _, files in _recursive_list(directory):
        for fname in sorted(files):
            for extension in white_list_formats:
                if fname.lower().endswith('.tiff'):
                    warnings.warn('Using \'.tiff\' files with multiple bands '
                                  'will cause distortion. '
                                  'Please verify your output.')
                if fname.lower().endswith('.' + extension):
                    yield root, fname


def _count_valid_files_in_directory(directory,
                                    white_list_formats,
                                    split,
                                    follow_links):
    """Counts files with extension in `white_list_formats` contained in `directory`.

    # Arguments
        directory: absolute path to the directory
            containing files to be counted
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
        split: tuple of floats (e.g. `(0.2, 0.6)`) to only take into
            account a certain fraction of files in each directory.
            E.g.: `segment=(0.6, 1.0)` would only account for last 40 percent
            of images in each directory.
        follow_links: boolean.

    # Returns
        the count of files with extension in `white_list_formats` contained in
        the directory.
    """
    num_files = len(list(
        _iter_valid_files(directory, white_list_formats, follow_links)))
    if split:
        start, stop = int(split[0] * num_files), int(split[1] * num_files)
    else:
        start, stop = 0, num_files
    return stop - start


def _list_valid_filenames_in_directory(directory, white_list_formats, split,
                                       class_indices, follow_links):
    """Lists paths of files in `subdir` with extensions in `white_list_formats`.

    # Arguments
        directory: absolute path to a directory containing the files to list.
            The directory name is used as class label
            and must be a key of `class_indices`.
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
        split: tuple of floats (e.g. `(0.2, 0.6)`) to only take into
            account a certain fraction of files in each directory.
            E.g.: `segment=(0.6, 1.0)` would only account for last 40 percent
            of images in each directory.
        class_indices: dictionary mapping a class name to its index.
        follow_links: boolean.

    # Returns
        classes: a list of class indices
        filenames: the path of valid files in `directory`, relative from
            `directory`'s parent (e.g., if `directory` is "dataset/class1",
            the filenames will be
            `["class1/file1.jpg", "class1/file2.jpg", ...]`).
    """
    dirname = os.path.basename(directory)
    if split:
        num_files = len(list(
            _iter_valid_files(directory, white_list_formats, follow_links)))
        start, stop = int(split[0] * num_files), int(split[1] * num_files)
        valid_files = list(
            _iter_valid_files(
                directory, white_list_formats, follow_links))[start: stop]
    else:
        valid_files = _iter_valid_files(
            directory, white_list_formats, follow_links)

    classes = []
    filenames = []
    for root, fname in valid_files:
        classes.append(class_indices[dirname])
        absolute_path = os.path.join(root, fname)
        relative_path = os.path.join(
            dirname, os.path.relpath(absolute_path, directory))
        filenames.append(relative_path)

    return classes, filenames

def _get_all_subfiles(directory, extension):
    testfilepaths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension) and not file.startswith('.'):
                testfilepaths.append(os.path.join(root, file))
    return testfilepaths


def count_dataset(datasetfiles):
    """
    A function to count the length of our entire dataset that can't be fit into ram,

    VERY INEFFICIENT. 

    TODO:
    - see if we can improve a way to determine the size of our entire dataset of chunks
    """
    from collections import OrderedDict

    datasetdict = OrderedDict()
    count = 0
    for dataset in datasetfiles:
        datastruct = np.load(dataset)
        ylabels = datastruct['ylabels']
        # have a dictionary store the indices between each dataset
        datasetdict[dataset] = [count,count+len(ylabels)]
        count += len(ylabels)

    return count, datasetdict

class CustomDirectoryIterator(Iterator):
    def __init__(self, directory, image_data_generator,
                 testname=None,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='binary',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png',
                 follow_links=False,
                 subset=None,
                 interpolation='nearest'):
    
        if data_format is None:
            data_format = K.image_data_format()
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)

        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'input', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input"'
                             ' or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation

        if subset is not None:
            validation_split = self.image_data_generator._validation_split
            if subset == 'validation':
                split = (0, validation_split)
            elif subset == 'training':
                split = (validation_split, 1)
            else:
                raise ValueError('Invalid subset name: ', subset,
                                 '; expected "training" or "validation"')
        else:
            split = None
        self.subset = subset

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp',
                              'ppm', 'tif', 'tiff'}
        # First, count the number of samples and classes.
        self.datasets = 0
        self.samples = 0

        if not classes:
            raise AttributeError("Need to pass in classes!")

        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        alldatasets = _get_all_subfiles(directory, extension='.npz')
        total_size_dataset, datasets_dict = count_dataset(alldatasets)
        # self.datasets = len(alldatasets)
        self.datasets = total_size_dataset
        self.datasets_dict = datasets_dict

        print('Found %d images belonging to %d classes.' %
              (self.datasets, self.num_classes))

        # Second, build an index of the images
        # in the different class subfolders.
        results = []
        self.filenames = alldatasets

        super(CustomDirectoryIterator, self).__init__(self.datasets,
                                                batch_size,
                                                shuffle,
                                                seed)

    def _sample_datasets(self, dataset_filelist, numsamps):
        """
        A function to help sample datasets. Assuming user has 
        multiple datasets passed into the directories, this can be used
        to sample a specific directory.


        """
        # get random indices to sample of the dataset
        randinds = np.random.choice(len(dataset_filelist), size=numsamps, replace=True)

        X = []
        Y = []
        # sample from here
        for i in range(randinds):
            fname = self.filenames[j]
            datastruct = np.load(fname)
            # print(datastruct.keys())
            # x = datastruct['image_tensor']
            x = datastruct['auxmats']
            y = datastruct['ylabels']

            # rand sample the dataset
            randsample = np.random.choice(len(y), size=1, replace=False)
            x = x[randsample,...]
            y = y[randsample,...]
            X.append(x)
            Y.append(y)

        return X, Y

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(
            (len(index_array),) + self.image_shape,
            dtype=K.floatx())
        batch_y = np.zeros(
            (len(index_array),),
            dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'

        # get random indices from file list
        # randinds = np.random.choice(self.datasets, size=len(index_array)//5, replace=True)
        # batch_filelist = np.array(self.filenames)[randinds]

        def _get_dataset_index(index):
            for f in self.filenames:
                ind_range = self.datasets_dict[f]
                if index >= ind_range[0] and index < ind_range[1]:
                    return f

        for i, j in enumerate(index_array):
            # obtain the filepath for this index and the corresponding index range
            fname = _get_dataset_index(j)
            ind_range = self.datasets_dict[fname]

            print("index range is: ", ind_range)
            print("j is ", j)
            # get actual index within this dataset by subtracting the lower bound
            j = j - ind_range[0]

            # load dataset
            datastruct = np.load(fname)
            # print(datastruct.keys())
            # x = datastruct['image_tensor']
            x = datastruct['auxmats']
            y = datastruct['ylabels']
            # print(x.shape)
            # print(y.shape)

            # get that index
            x = x[j, ...].squeeze().reshape(self.image_shape)
            y = y[j, ...].squeeze()

            params = self.image_data_generator.get_random_transform(x.shape)
            x = self.image_data_generator.apply_transform(x, params)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
            batch_y[i] = y

        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'binary':
            batch_y = batch_y.astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros(
                (len(batch_x), self.num_classes),
                dtype=K.floatx())
            for i, label in enumerate(batch_y):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)