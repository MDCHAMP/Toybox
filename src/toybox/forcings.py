import numpy as np
class excitation():
    '''
    Bass class for excitations implemented in toybox, contains generic input filtering and conditioning methods
    '''    
    # def low_pass(self, cutoff):
    #     pass

    # def high_pass(self, cutoff):
    #     pass

    # def band_pass(self, cut_low, cut_high):
    #     pass

    # def _apply_filters(self):
    #     pass

# %% Excitations
class white_gaussian(excitation):

    def __init__(self, u, sig):
        self.u = u
        self.sig = sig

    def generate(self, ts):
        n = len(ts)
        return np.random.normal(self.u, self.sig, size=(n, 1))

class sinusoid(excitation):
    
    def __init__(self, a, w, phi=0):
        self.a = a
        self.w = w
        self.phi = phi

    def generate(self, ts):
        return self.a * np.sin(self.w*ts - self.phi)


class impact(excitation):
    pass

# %% MDOF multi-DOF excitations 

class shaker():

    def __init__(self, excitations):
        self.excitations = excitations
    
    def generate(self, ts):
        '''
        Generate a time series by parsing the         
        '''
        out = np.zeros((len(self.excitations), len(ts)))
        for i, e in enumerate(self.excitations):
            if isinstance(e, excitation): # Case: toybox excitation implementation
                out[i,:] = np.squeeze(e.generate(ts))
            elif (e is not None) and (len(e)==len(ts)):  # Case: custom time series with appropriate length
                out[i,:] = e
            else:  # Case: all other inputs are interpreted as no forcing
                out[i,:] = np.zeros(len(ts))
        return out.T
    
    def get_x_t(self, ts, xs):
        '''
        Wrap a function to return the forcing at time t
        '''        
        ndofs = len(self.excitations)
        def x_t(t):
            '''
            Interpolate for the forcing between timesteps
            '''
            x = np.zeros((ndofs, ))
            for dof in range(ndofs):
                x[dof] = np.interp(t, ts, xs[:,dof])
            return x
        return x_t
