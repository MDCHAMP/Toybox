'''
Commonly used systems instantiated for ease of use 
'''

import numpy as np

import toybox.nonlinearities as nonlinear
from toybox.base import system


# %% Generic subclasses

class symetric(system):
    '''
    Symmetric MDOF system aka tri-diagonal system.
    '''
    name = 'Symmetric MDOF system'

    def __init__(self, dofs=1, m=1, c=0.1, k=10, N=None):
        if dofs == 1:
            M = np.array([[m]])
            C = np.array([[c]])
            K = np.array([[k]])
        else:
            square = np.zeros((dofs, dofs))
            diag = np.array([1.0]*(dofs))
            off_diag = np.array([-1.0]*(dofs-1))
            M = square + np.diag(diag)
            C = square + np.diag(2*c*diag) + \
                np.diag(c*off_diag, 1) + np.diag(c*off_diag, -1)
            K = square + np.diag(2*k*diag) + \
                np.diag(k*off_diag, 1) + np.diag(k*off_diag, -1)
        super().__init__(M, C, K, N)


