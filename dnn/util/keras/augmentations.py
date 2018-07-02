import numpy as np 

class Augmentations(object):
    @staticmethod
    def preprocess_imgwithnoise(image_tensor):
        # preprocessing_function: function that will be implied on each input.
        #         The function will run before any other modification on it.
        #         The function should take one argument:
        #         one image (Numpy tensor with rank 3),
        #         and should output a Numpy tensor with the same shape.
        # assert image_tensor.shape[1] == image_tensor.shape[2]
        stdmult = 0.1
        length_imsize = image_tensor.shape[0]
        width_imsize = image_tensor.shape[1]
        numchans = image_tensor.shape[2]
        for i in range(numchans):
            feat = image_tensor[..., i].ravel()
            noise_add = np.random.normal(
                scale=stdmult*np.std(feat), size=feat.size).reshape(length_imsize, width_imsize)
            image_tensor[..., i] = image_tensor[..., i] + noise_add
        return image_tensor