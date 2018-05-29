import math as m
import numpy as np
# np.random.seed(123)
import scipy.io
from sklearn.decomposition import PCA
from scipy.interpolate import griddata
from sklearn.preprocessing import scale


class DataGenerator(object):
    def __init__(self, dim_x=32, dim_y=32, dim_z=32, batch_size=32, shuffle=True):
        import keras

        'Initialization'
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __get_exploration_order(self, list_IDs):
        'Generates order of exploration'
        # Find exploration order
        indexes = np.arange(len(list_IDs))
        if self.shuffle == True:
            np.random.shuffle(indexes)
        return indexes

    # def __data_generation(self, labels, list_IDs_temp):
    #     'Generates data of batch_size samples'
    #     # X : (n_samples, v_size, v_size, v_size, n_channels)
    #     # Initialization
    #     X = np.empty((self.batch_size, self.dim_x, self.dim_y, self.dim_z, 1))
    #     y = np.empty((self.batch_size), dtype = int)

    #     # Generate data
    #     for i, ID in enumerate(list_IDs_temp):
    #       # Store volume
    #       X[i, :, :, :, 0] = np.load(ID + '.npy')

    #       # Store class
    #       y[i] = labels[ID]

    #     return X, y

    def __load_data(self, file_ID):
        data = np.load(file_ID)
        images = data['image_tensor']
        metadata = data['metadata'].item()

        # load the ylabeled data
        ylabels = metadata['ylabels']
        invert_y = 1 - ylabels
        y = np.concatenate((ylabels, invert_y), axis=1)

        # images = normalizeimages(images) # normalize the images for each frequency band
        # assert the shape of the images
        assert images.shape[2] == images.shape[3]
        assert images.shape[2] == self.dim_x
        assert images.shape[1] == self.dim_z
        images = images.swapaxes(1, 3)
        X = images.astype("float32")
        return X, y

    # def generate(self, labels, list_IDs):
    #     'Generates batches of samples'
    #     # Infinite loop
    #     while 1:
    #       # Generate order of exploration of dataset
    #       indexes = self.__get_exploration_order(list_IDs)

    #       # Generate batches
    #       imax = int(len(indexes)/self.batch_size)
    #       for i in range(imax):
    #           # Find list of IDs
    #           list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

    #           # Generate data
    #           X, y = self.__data_generation(labels, list_IDs_temp)

    #           yield X, y

    def generate_fromdir(self, list_files):
        'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(list_files)

            for i in range(len(indexes)):
                # get current file to load up
                file_ID = list_files[i]

                X, y = self.__load_data(file_ID)

                # Generate batches on this file's data
                imax = int(len(y)/self.batch_size)
                data_indices = self.__get_exploration_order(range(len(y)))
                for j in range(imax):
                    # Find list of indices through the data
                    indices = data_indices[j *
                                           self.batch_size:(j+1)*self.batch_size]

                    batchX = X[indices, ...]
                    batchy = y[indices]

                    yield batchX, batchy

    def standardize(self, x):
        """Apply the normalization configuration to a batch of inputs.
        # Arguments
            x: batch of inputs to be normalized.
        # Returns
            The inputs, normalized.
        """
        if self.samplewise_center:
            x -= np.mean(x, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, keepdims=True) + keras.epsilon())

        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_center`, but it hasn\'t '
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= (self.std + K.epsilon())
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_std_normalization`, but it hasn\'t '
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        return x

    def fit(self, x,
            augment=False,
            rounds=1,
            seed=None):
        """Fits internal statistics to some sample data.
        Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.
        # Arguments
            x: Numpy array, the data to fit on. Should have rank 4.
                In case of grayscale data,
                the channels axis should have value 1, and in case
                of RGB data, it should have value 3.
            augment: Whether to fit on randomly augmented samples
            rounds: If `augment`,
                how many augmentation passes to do over the data
            seed: random seed.
        # Raises
            ValueError: in case of invalid input `x`.
        """
        x = np.asarray(x, dtype=keras.floatx())
        if seed is not None:
            np.random.seed(seed)

        x = np.copy(x)
        if augment:
            ax = np.zeros(tuple([rounds * x.shape[0]] +
                                list(x.shape)[1:]), dtype=K.floatx())
            for r in range(rounds):
                for i in range(x.shape[0]):
                    ax[i + r * x.shape[0]] = self.random_transform(x[i])
            x = ax

        if self.featurewise_center:
            self.mean = np.mean(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.mean = np.reshape(self.mean, broadcast_shape)
            x -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.std = np.reshape(self.std, broadcast_shape)
            x /= (self.std + K.epsilon())


'''
A Suite of utility functions for preprocessing wrapped within
a class
'''


class DataHandler(object):
    def __init__(self, data=None, labels=None):
        self.data = data
        self.labels = labels

    def reformatinput(self, data: np.ndarray, indices: list):
        '''
        Receives the the indices for train and test datasets.
        Outputs the train, validation, and test data and label datasets.

        Parameters:
        data            (np.ndarray) of [n_samples, n_colors, W, H], or
                        [n_timewindows, n_samples, n_colors, W, H] for time dependencies
        indices         (list of tuples) of indice tuples to include in training,
                        validation, testing.
                        indices[0] = train
                        indices[1] = test

        Output:
        (list) of tuples for training, validation, testing 
        '''
        # get train and test indices
        trainIndices = indices[0][len(indices[1]):].astype(np.int32)
        validIndices = indices[0][:len(indices[1])].astype(
            np.int32)  # use part of training for validation
        testIndices = indices[1].astype(np.int32)

        # gets train, valid, test labels as int32
        trainlabels = np.squeeze(indices[trainIndices]).astype(np.int32)
        validlabels = np.squeeze(indices[validIndices]).astype(np.int32)
        testlabels = np.squeeze(indices[testIndices]).astype(np.int32)

        # Shuffling training data
        # shuffledIndices = np.random.permutation(len(trainIndices))
        # trainIndices = trainIndices[shuffledIndices]

        # get the data tuples for train, valid, test by slicing thru n_samples
        if data.ndim == 4:
            return [(data[trainIndices], trainlabels),
                    (data[validIndices], validlabels),
                    (data[testIndices], testlabels)]
        elif data.ndim == 5:
            return [(data[:, trainIndices], trainlabels),
                    (data[:, validIndices], validlabels),
                    (data[:, testIndices], testlabels)]

    def load_mat_data(self, data_file: str):
        '''
        Loads the data from MAT file. MAT file should contain two
        variables. 'featMat' which contains the feature matrix in the
        shape of [samples, features] and 'labels' which contains the output
        labels as a vector. Label numbers are assumed to start from 1.

        Parameters
        ----------
        data_file       (str) for the fullpath to the file

        Returns
        -------
        data: array_like
        '''
        print("Loading data from %s" % (data_file))

        dataMat = scipy.io.loadmat(data_file, mat_dtype=True)

        print("Data loading complete. Shape is %r" %
              (dataMat['featMat'].shape,))
        # Sequential indices
        return dataMat['features'][:, :-1], dataMat['features'][:, -1] - 1

    def load_mat_locs(self, datafile: str):
        # '../Sample data/Neuroscan_locs_orig.mat'
        locs = scipy.io.loadmat(datafile)
        return locs

    def cart2sph(self, x: float, y: float, z: float):
        '''
        Transform Cartesian coordinates to spherical

        Paramters:
        x           (float) X coordinate
        y           (float) Y coordinate
        z           (float) Z coordinate

        :return: radius, elevation, azimuth
        '''
        x2_y2 = x**2 + y**2
        r = m.sqrt(x2_y2 + z**2)                    # r
        elev = m.atan2(z, m.sqrt(x2_y2))            # Elevation
        az = m.atan2(y, x)                          # Azimuth
        return r, elev, az

    def pol2cart(self, theta: float, rho: float):
        '''
        Transform polar coordinates to Cartesian

        Parameters
        theta          (float) angle value
        rho            (float) radius value

        :return: X, Y
        '''
        return rho * m.cos(theta), rho * m.sin(theta)

    def azim_proj(self, pos: list):
        '''
        Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
        Imagine a plane being placed against (tangent to) a globe. If
        a light source inside the globe projects the graticule onto
        the plane the result would be a planar, or azimuthal, map
        projection.

        Parameters:
        pos         (list) positions in 3D Cartesian coordinates

        :return: projected coordinates using Azimuthal Equidistant Projection
        '''
        [r, elev, az] = self.cart2sph(pos[0], pos[1], pos[2])
        return self.pol2cart(az, m.pi / 2 - elev)

    def augment_EEG(self, data: np.ndarray, stdMult: float=0.1, pca: bool=False, n_components: int=2):
        '''
        Augment data by adding normal noise to each feature.

        Parameters:
        data            (np.ndarray) EEG feature data as a matrix 
                        (n_samples x n_features)
        stdMult         (float) Multiplier for std of added noise
        pca             (bool) if True will perform PCA on data and add noise proportional to PCA components.
        n_components    (int) Number of components to consider when using PCA.

        :return: Augmented data as a matrix (n_samples x n_features)
        '''
        augData = np.zeros(data.shape)
        if pca:
            pca = PCA(n_components=n_components)
            pca.fit(data)
            components = pca.components_
            variances = pca.explained_variance_ratio_
            coeffs = np.random.normal(
                scale=stdMult, size=pca.n_components) * variances
            for s, sample in enumerate(data):
                augData[s, :] = sample + \
                    (components * coeffs.reshape((n_components, -1))).sum(axis=0)
        else:
            # Add Gaussian noise with std determined by weighted std of each feature
            for f, feat in enumerate(data.transpose()):
                augData[:, f] = feat + \
                    np.random.normal(
                        scale=stdMult*np.std(feat), size=feat.size)
        return augData

    def gen_images(self, locs: np.ndarray, feature_tensor: np.ndarray, n_gridpoints: int=32, normalize: bool=True,
                   augment: bool=False, pca: bool=False, std_mult: float=0.1, n_components: int=2, edgeless: bool=False):
        '''
        Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode

        Parameters
        locs                (np.ndarray) An array with shape [n_electrodes, 2] containing X, Y
                            coordinates for each electrode.
        features            (np.ndarray) Feature matrix as [n_samples, n_features]
                            as [numchans, numfeatures, numsamples]
                            Features are as columns.
                            Features corresponding to each frequency band are concatenated.
                            (alpha1, alpha2, ..., beta1, beta2,...)
        n_gridpoints        (int) Number of pixels in the output images
        normalize           (bool) Flag for whether to normalize each band over all samples (default=True)
        augment             (bool) Flag for generating augmented images (default=False)
        pca                 (bool) Flag for PCA based data augmentation (default=False)
        std_mult            (float) Multiplier for std of added noise
        n_components        (int) Number of components in PCA to retain for augmentation
        edgeless            (bool) If True generates edgeless images by adding artificial channels
                            at four corners of the image with value = 0 (default=False).

        :return:            Tensor of size [samples, colors, W, H] containing generated
                            images.
        '''
        feat_array_temp = []            # list holder for feature array
        temp_interp = []

        numcontacts = feature_tensor.shape[0]     # Number of electrodes
        n_colors = feature_tensor.shape[1]
        # n_colors = 4
        numsamples = feature_tensor.shape[2]

        # Interpolate the values into a grid of x/y coords
        grid_x, grid_y = np.mgrid[min(locs[:, 0]):max(locs[:, 0]):n_gridpoints*1j,
                                  min(locs[:, 1]):max(locs[:, 1]):n_gridpoints*1j]

        # loop through each color
        for c in range(n_colors):
            # build feature array from [ncontacts, 1 freq band, nsamples] squeezed and swapped axes
            feat_array_temp.append(
                feature_tensor[:, c, :].squeeze().swapaxes(0, 1))

            if c == 0:
                print(feat_array_temp[0].shape)

            if augment:  # add data augmentation -> either pca or not
                feat_array_temp[c] = self.augment_EEG(
                    feat_array_temp[c], std_mult, pca=pca, n_components=n_components)

            # build temporary interpolator matrix
            temp_interp.append(
                np.zeros([numsamples, n_gridpoints, n_gridpoints]))
        # Generate edgeless images -> add 4 locations (minx,miny),...,(maxx,maxy)
        if edgeless:
            min_x, min_y = np.min(locs, axis=0)
            max_x, max_y = np.max(locs, axis=0)
            locs = np.append(locs, np.array([[min_x, min_y],
                                             [min_x, max_y],
                                             [max_x, min_y],
                                             [max_x, max_y]]), axis=0)
            for c in range(n_colors):
                feat_array_temp[c] = np.append(
                    feat_array_temp[c], np.zeros((numsamples, 4)), axis=1)

       # Interpolating for all samples across all features
        for i in range(numsamples):
            for c in range(n_colors):
                temp_interp[c][i, :, :] = griddata(points=locs,
                                                   values=feat_array_temp[c][i, :],
                                                   xi=(grid_x, grid_y),
                                                   method='cubic',
                                                   fill_value=np.nan)
            print('Interpolating {0}/{1}\r'.format(i+1, numsamples), end='\r')

        # Normalize every color (freq band) range of values
        for c in range(n_colors):
            if normalize:
                temp_interp[c][~np.isnan(temp_interp[c])] = scale(
                    X=temp_interp[c][~np.isnan(temp_interp[c])])
            # convert all nans to 0
            temp_interp[c] = np.nan_to_num(temp_interp[c])

        # swap axes to have [samples, colors, W, H]
        return np.swapaxes(np.asarray(temp_interp), 0, 1)

    def gen_images3d(self, locs: np.ndarray,
                     feature_tensor: np.ndarray,
                     n_gridpoints: int=32,
                     normalize: bool=True,
                     augment: bool=False,
                     std_mult: float=0.1,
                     edgeless: bool=False):
        '''
        Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode

        Parameters
        locs                (np.ndarray) An array with shape [n_electrodes, 2] containing X, Y
                            coordinates for each electrode.
        features            (np.ndarray) Feature matrix as [n_samples, n_features]
                            as [numchans, numfeatures, numsamples]
                            Features are as columns.
                            Features corresponding to each frequency band are concatenated.
                            (alpha1, alpha2, ..., beta1, beta2,...)
        n_gridpoints        (int) Number of pixels in the output images
        normalize           (bool) Flag for whether to normalize each band over all samples (default=True)
        augment             (bool) Flag for generating augmented images (default=False)
        std_mult            (float) Multiplier for std of added noise
        edgeless            (bool) If True generates edgeless images by adding artificial channels
                            at four corners of the image with value = 0 (default=False).

        :return:            Tensor of size [samples, colors, W, H] containing generated
                            images.
        '''
        feat_array_temp = []            # list holder for feature array
        temp_interp = []

        numcontacts, n_colors, numsamples = feature_tensor.shape     # Number of electrodes

        # Interpolate the values into a grid of x/y coords
        grid_x, grid_y, grid_z = np.mgrid[min(locs[:, 0]):max(locs[:, 0]):n_gridpoints*1j,  # x
                                          # y
                                          min(locs[:, 1]):max(locs[:, 1]):n_gridpoints*1j,
                                          # z
                                          min(locs[:, 2]):max(locs[:, 2]):n_gridpoints*1j,
                                          ]

        # loop through each color
        for c in range(n_colors):
            # build feature array from [ncontacts, 1 freq band, nsamples] squeezed and swapped axes
            feat_array_temp.append(
                feature_tensor[:, c, :].squeeze().swapaxes(0, 1))

            if c == 0:
                print(feat_array_temp[0].shape)

            if augment:  # add data augmentation -> either pca or not
                feat_array_temp[c] = self.augment_EEG(
                    feat_array_temp[c], std_mult, n_components=n_components)

            # build temporary interpolator matrix
            temp_interp.append(
                np.zeros([numsamples, n_gridpoints, n_gridpoints, n_gridpoints]))
        # Generate edgeless images -> add 4 locations (minx,miny),...,(maxx,maxy)
        if edgeless:
            min_x, min_y, min_z = np.min(locs, axis=0)
            max_x, max_y, max_z = np.max(locs, axis=0)
            locs = np.append(locs, np.array([[min_x, min_y, min_z],
                                             [min_x, max_y, min_z],
                                             [max_x, min_y, min_z],
                                             [max_x, max_y, min_z],
                                             [min_x, min_y, max_z],
                                             [min_x, max_y, max_z],
                                             [max_x, min_y, max_z],
                                             [max_x, max_y, max_z]]), axis=0)
            for c in range(n_colors):
                feat_array_temp[c] = np.append(
                    feat_array_temp[c], np.zeros((numsamples, 4)), axis=1)

       # Interpolating for all samples across all features
        for i in range(numsamples):
            for c in range(n_colors):
                temp_interp[c][i, :, :, :] = griddata(points=locs,
                                                      values=feat_array_temp[c][i, :],
                                                      xi=(grid_x, grid_y,
                                                          grid_z),
                                                      method='linear',
                                                      fill_value=np.nan)
            print('Interpolating {0}/{1}\r'.format(i+1, numsamples), end='\r')

        # Normalize every color (freq band) range of values
        for c in range(n_colors):
            if normalize:
                temp_interp[c][~np.isnan(temp_interp[c])] = scale(
                    X=temp_interp[c][~np.isnan(temp_interp[c])])
            # convert all nans to 0
            temp_interp[c] = np.nan_to_num(temp_interp[c])

        # swap axes to have [samples, colors, W, H]
        return np.swapaxes(np.asarray(temp_interp), 0, 1)

    def computelabels(self, seizonsets, seizoffsets, timepoints):
        if not isinstance(seizonsets, list) and not isinstance(seizonsets, np.ndarray):
            seizonsets = np.array([seizonsets])
        if not isinstance(seizoffsets, list) and not isinstance(seizoffsets, np.ndarray):
            seizoffsets = np.array([seizoffsets])

        ylabels = np.zeros((timepoints.shape[0], 1))

        if len(seizonsets) == 0 or seizonsets[0] == np.nan:
            print('no seizure times in <computelabels>!')
            return -1
        if len(seizoffsets) == 0:
            # Determine the starting window point of the seiztimes
            start_position = np.where(timepoints[:, 1] > seizonsets[0])[0][0]
            ylabels[start_position:] = 1
            return ylabels

        for idx in range(len(seizoffsets)):
            # Determine the starting window point of the seiztimes
            start_position = np.where(timepoints[:, 1] > seizonsets[idx])[0][0]

            # Determine the starting window point of the seiztimes
            end_position = np.where(timepoints[:, 1] > seizoffsets[idx])[0][0]

            # print(start_position, end_position)
            ylabels[start_position:end_position] = 1

        if len(seizonsets) > idx + 1:
            # Determine the starting window point of the seiztimes
            start_position = np.where(
                timepoints[:, 1] > seizonsets[idx+1])[0][0]
            ylabels[start_position:] = 1

        return ylabels
