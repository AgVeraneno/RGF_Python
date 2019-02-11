import numpy as np
#import cupy as cp

class Material():
    def __init__(self, mat_name):
        '''
        Physics const.
        '''
        self.q = 1.6e-19                      # C. electron charge
        self.me = 9.11e-31                    # Kg. electron rest mass
        self.h_bar = 1.05457e-34              # J*s. Planck's const. divided by 2*pi
        self.kB = 1.38e-23                    # J/K. Boltzmann const
        self.E_flux = 1.23984188e-6           # V*m
        '''
        material parameters
        '''
        self.name = mat_name
        self.__parameters__()
    def __parameters__(self):
        if self.name == 'Graphene':
            '''
            Graphene const.
            '''
            self.a = 1.42e-10                     # m. same atom's nearest neighbor distance
            self.ax = 3*self.a                    # m. unit cell width
            self.acc = self.a/3**0.5              # m. carbon to carbon distance
            self.K_norm = 4/3*np.pi/self.acc      # m-1. normalized K vector
            self.vF = 8e5                         # m/s. Fermi velocity for graphene
            ### BLG const.
            self.r0 = 2.8                         # J. A1-B1 hopping energy
            self.r1 = 0.39                        # J. A2-B1 hopping energy
            self.r3 = 0.315                       # J. A1-B2 hopping energy
        else:
            raise ValueError('Material "',self.name,'" database is not bulit.')