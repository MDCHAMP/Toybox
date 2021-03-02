import numpy as np

# %% Excitations

class excitation():
    pass


class white_gaussian(excitation):

    def __init__(self, u, sig):
        self.u = u
        self.sig = sig

    def generate(self, n, d):
        return np.zeros((n,d))


class sinusoid(excitation):
    pass


class impact(excitation):
    pass


# %% MDOF multi-DOF excitations 

class shaker():

    def __init__(self, excitations):
        self.excitations = excitations
    
    def generate(self, n):
        out = np.zeros((n, len(self.excitations)))
        for i, e in enumerate(self.excitations):
            if isinstance(e, excitation):
                out[i,:] = e.generate(n)
            else: 
                out[i,:] = np.zeros(n)
        return out

