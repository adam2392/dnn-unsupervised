import numpy as np 
import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

from dnn.util.generators.base.baseaux import AuxNumpyArrayIterator

class AuxImgDataGenerator(ImageDataGenerator):
    def __init__(self, *args, **kwargs):
        super(AuxImgDataGenerator, self).__init__(*args, **kwargs)
        self.iterator=None

    def flow(self, x, y=None, 
            batch_size=32, shuffle=True,
            sample_weight=None, seed=None,
            save_to_dir=None, save_prefix='', 
            save_format='png', subset=None):
        """Takes data & label arrays, generates batches of augmented data.

        # Arguments
            x: Input data. Numpy array of rank 4 or a tuple.
                If tuple, the first element
                should contain the images and the second element
                another numpy array or a list of numpy arrays
                that gets passed to the output
                without any modifications.
                Can be used to feed the model miscellaneous data
                along with the images.
                In case of grayscale data, the channels axis of the image array
                should have value 1, and in case
                of RGB data, it should have value 3.
            y: Labels.
            batch_size: Int (default: 32).
            shuffle: Boolean (default: True).
            sample_weight: Sample weights.
            seed: Int (default: None).
            save_to_dir: None or str (default: None).
                This allows you to optionally specify a directory
                to which to save the augmented pictures being generated
                (useful for visualizing what you are doing).
            save_prefix: Str (default: `''`).
                Prefix to use for filenames of saved pictures
                (only relevant if `save_to_dir` is set).
                save_format: one of "png", "jpeg"
                (only relevant if `save_to_dir` is set). Default: "png".
            subset: Subset of data (`"training"` or `"validation"`) if
                `validation_split` is set in `ImageDataGenerator`.

        # Returns
            An `Iterator` yielding tuples of `(x, y)`
                where `x` is a numpy array of image data
                (in the case of a single image input) or a list
                of numpy arrays (in the case with
                additional inputs) and `y` is a numpy array
                of corresponding labels. If 'sample_weight' is not None,
                the yielded tuples are of the form `(x, y, sample_weight)`.
                If `y` is None, only the numpy array `x` is returned.
        """
        return AuxNumpyArrayIterator(x, y, 
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            sample_weight=sample_weight,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            subset=subset)

    # def fit(self, x, augment=False, rounds=1, seed=None):
    #     """Fits the data generator to some sample data.

    #     This computes the internal data stats related to the
    #     data-dependent transformations, based on an array of sample data.

    #     Only required if `featurewise_center` or
    #     `featurewise_std_normalization` or `zca_whitening` are set to True.

    #     # Arguments
    #         x: Sample data. Should have rank 4.
    #          In case of grayscale data,
    #          the channels axis should have value 1, and in case
    #          of RGB data, it should have value 3.
    #         augment: Boolean (default: False).
    #             Whether to fit on randomly augmented samples.
    #         rounds: Int (default: 1).
    #             If using data augmentation (`augment=True`),
    #             this is how many augmentation passes over the data to use.
    #         seed: Int (default: None). Random seed.
    #     """
    #     x = np.asarray(x, dtype=K.floatx())

    #     if x.ndim == 4:
    #         # reshape to a list of images
    #         imsize = x.shape[-2]
    #         chansize = x.shape[-1]
    #         x = x.reshape(-1, imsize, imsize, chansize)

    #     if x.ndim != 4:
    #         raise ValueError('Input to `.fit()` should have rank 4. '
    #                          'Got array with shape: ' + str(x.shape))
    #     if x.shape[self.channel_axis] not in {1, 3, 4}:
    #         warnings.warn(
    #             'Expected input to be images (as Numpy array) '
    #             'following the data format convention "' +
    #             self.data_format + '" (channels on axis ' +
    #             str(self.channel_axis) + '), i.e. expected '
    #             'either 1, 3 or 4 channels on axis ' +
    #             str(self.channel_axis) + '. '
    #             'However, it was passed an array with shape ' +
    #             str(x.shape) + ' (' + str(x.shape[self.channel_axis]) +
    #             ' channels).')

    #     if seed is not None:
    #         np.random.seed(seed)

    #     x = np.copy(x)
    #     if augment:
    #         ax = np.zeros(
    #             tuple([rounds * x.shape[0]] + list(x.shape)[1:]),
    #             dtype=K.floatx())
    #         for r in range(rounds):
    #             for i in range(x.shape[0]):
    #                 ax[i + r * x.shape[0]] = self.random_transform(x[i])
    #         x = ax

    #     if self.featurewise_center:
    #         self.mean = np.mean(x, axis=(0, self.row_axis, self.col_axis))
    #         broadcast_shape = [1, 1, 1]
    #         broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
    #         self.mean = np.reshape(self.mean, broadcast_shape)
    #         x -= self.mean

    #     if self.featurewise_std_normalization:
    #         self.std = np.std(x, axis=(0, self.row_axis, self.col_axis))
    #         broadcast_shape = [1, 1, 1]
    #         broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
    #         self.std = np.reshape(self.std, broadcast_shape)
    #         x /= (self.std + K.epsilon())

    #     if self.zca_whitening:
    #         flat_x = np.reshape(
    #             x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
    #         sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
    #         u, s, _ = linalg.svd(sigma)
    #         s_inv = 1. / np.sqrt(s[np.newaxis] + self.zca_epsilon)
    #         self.principal_components = (u * s_inv).dot(u.T)

    # 