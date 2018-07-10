from itertools import product
import functools
from keras.losses import binary_crossentropy
def w_categorical_crossentropy(y_true, y_pred, weights):
    ''' https://github.com/keras-team/keras/issues/2115
        https://stackoverflow.com/questions/46202839/weight-different-misclassifications-differently-keras

     '''
    weights = np.array(weights)
    nb_cl = weights.shape[0]
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    
    final_mask += (weights * y_pred_max_mat[:, :] * y_true[:, :])
    return K.categorical_crossentropy(y_pred, y_true) * final_mask

def weighted_binary_crossentropy(y_true, y_pred):
    false_positive_weight = 2       
    false_negative_weight = 1
    thresh = 0.5
    y_pred_true = K.greater_equal(thresh,y_pred)
    y_not_true = K.less_equal(thresh,y_true)
    false_positive_tensor = K.equal(y_pred_true,y_not_true)

    #changing from here

    #first let's transform the bool tensor in numbers - maybe you need float64 depending on your configuration
    false_positive_tensor = K.cast(false_positive_tensor,'float32') 

    #and let's create it's complement (the non false positives)
    complement = 1 - false_positive_tensor

    #now we're going to separate two groups
    falsePosGroupTrue = y_true * false_positive_tensor
    falsePosGroupPred = y_pred * false_positive_tensor

    nonFalseGroupTrue = y_true * complement
    nonFalseGroupPred = y_pred * complement

    #let's calculate one crossentropy loss for each group
    #(directly from the keras loss functions imported above)
    falsePosLoss = binary_crossentropy(falsePosGroupTrue,falsePosGroupPred)
    nonFalseLoss = binary_crossentropy(nonFalseGroupTrue,nonFalseGroupPred)

    #return them weighted:
    return (false_positive_weight*falsePosLoss) + (nonFalseLoss)
