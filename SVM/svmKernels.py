"""
Custom SVM Kernels

Author: Eric Eaton, 2014

"""

import numpy as np

_polyDegree = 2
_gaussSigma = 1


def myPolynomialKernel(X1, X2):
    '''
        Arguments:  
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    return (np.dot(X1, np.transpose(X2)) + 1)**_polyDegree


def myGaussianKernel(X1, X2):
    '''
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    n = X1.shape[0]
    m = X2.shape[0]
    ans = np.matrix(np.zeros([n, m]))
    for i in range(0,n):
        dist = X2-X1[i]
        norm =np.sum(dist**2, axis=1) 
        ans[i]= np.exp(-norm/(2*_gaussSigma**2))    

    return ans

