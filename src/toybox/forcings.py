import numpy as np

# %% Excitations

class excitation():
    pass


class white_gaussian(excitation):

    def __init__(self, u, sig):
        self.u = u
        self.sig = sig

    def generate(self, n):
        return np.random.normal(self.u, self.sig, size=(n, 1))


class sinusoid(excitation):
    pass


class impact(excitation):
    pass


# %% MDOF multi-DOF excitations 

class shaker():

    def __init__(self, excitations):
        self.excitations = excitations
    
    def generate(self, n):
        out = np.zeros((len(self.excitations), n))
        for i, e in enumerate(self.excitations):
            if isinstance(e, excitation):
                out[i,:] = np.squeeze(e.generate(n))
            elif False:
                # TODO handle custom forcing here  
                out[i,:] = e
            else: 
                out[i,:] = np.zeros(n)
        return out.T

