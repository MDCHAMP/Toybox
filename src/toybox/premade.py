'''
Commonly used systems instantiated for ease of use 
'''

import numpy as np

import toybox.nonlinearities as nonlinear
from toybox.base import system


# %% Generic subclasses

class symetric(system):
    '''
    Symmetric MDOF system
    '''
    name = 'Symmetric MDOF system'

    def __init__(self, dofs, m=1, c=0.1, k=10, N=None):
        square = np.zeros((dofs, dofs))
        diag = np.array([1.0]*(dofs))
        off_diag = np.array([-1.0]*(dofs-1))

        self.M = square + np.diag(diag)
        self.C = self.M*(c/m) + np.diag(2*c*diag) + \
            np.diag(c*off_diag, 1) + np.diag(off_diag, -1)
        self.K = self.M*(k/m) + np.diag(2*k*diag) + \
            np.diag(k*off_diag, 1) + np.diag(k*off_diag, -1)
        self.N = N
        self.dofs = self.M.shape[0]
        self.slopes = None


# %% Some simple SDOFS


class underdamped_SDOF(symetric):
    '''
    Underdamped linear
    '''
    name = 'Underdamped linear'

    def __init__(self):
        super().__init__(1, 1, 0.1, 100)


class criticalydamped_SDOF(symetric):
    '''
    Critically damped linear    
    '''
    name = 'Underdamped linear'

    def __init__(self):
        super().__init__(1, 1, 20, 100)


class overdamped_SDOF(symetric):
    '''
    Overdamped linear
    '''
    name = 'Underdamped linear'

    def __init__(self):
        super().__init__(1, 1, 100, 100)


class duffing_SDOF(symetric):
    '''
    Duffing equation
    '''
    name = 'Duffing equation'

    def __init__(self, k3n=1500):
        super().__init__(1, 1, 0.1, 100)
        self.N = nonlinear.cubic_stifness(1, k3n)
