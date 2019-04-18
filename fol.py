"""

First Order Logic (FOL) rules

"""

import warnings
import numpy as np
import tensorflow as tf 

#----------------------------------------------------
# BUT rule
#----------------------------------------------------

class FOL_But(object):
    """ x=x1_but_x2 => { y => pred(x2) AND pred(x2) => y } """
    def __init__(self, K, input, fea):
        """ Initialize
    
    :type K: int
    :param K: the number of classes 

    :type fea: theano.tensor.dtensor4
    :param fea: symbolic feature tensor, of shape 3
                fea[0]   : 1 if x=x1_but_x2, 0 otherwise
                fea[1:2] : classifier.predict_p(x_2)
    """
        assert K == 2
        self.input = input
        self.fea = fea


    """
    Efficient version specific to the BUT-rule

    """
    def log_distribution(self, w, X=None, F=None):
        if F == None:
            X, F = self.input, self.fea
        F_mask = F[:,0] 
        F_fea = F[:,1:]
        # y = 0
        distr_y0 = w*F_mask*F_fea[:,0]
        # y = 1 
        distr_y1 = w*F_mask*F_fea[:,1]
        distr_y0 = distr_y0.reshape([distr_y0.shape[0],1])
        distr_y1 = np.exp(distr_y1.reshape([distr_y1.shape[0],1]))
        distr = np.concatenate((distr_y0, distr_y1),axis=1)
        return distr



