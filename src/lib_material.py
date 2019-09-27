import numpy as np

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
        self.__parameters__(mat_name)
    def __parameters__(self):
        if mat_name == 'Graphene':
            '''
            Graphene const.
            '''
            self.a = 2.46e-10                     # m. same atom's nearest neighbor distance
            self.acc = self.a/3**0.5              # m. carbon to carbon distance
            self.ax = 3*self.acc                  # m. unit cell width
            self.K_norm = 4/3*np.pi/self.acc      # m-1. normalized K vector
            self.vF = 8e5                         # m/s. Fermi velocity for graphene
            ### hopping energy
            self.r0 = 2.8                         # J. A1-B1 hopping energy
            self.r1 = 0.39                        # J. A2-B1 hopping energy
            self.r3 = 0.315                       # J. A1-B2 hopping energy
        elif mat_name == 'WSe2':
            '''
            WSe2 const.
            '''
            self.acc = 3.125e-10                  # m. nearest neighbor distance
            self.a = self.acc*3**0.5                # m. carbon to carbon distance
            self.ax = 3*self.acc                    # m. unit cell width
            self.K_norm = 4/3*np.pi/self.acc      # m-1. normalized K vector
            ### hopping energy
            self.A0=np.zeros((6,6),dtype=np.complex128)
            self.A0[0,0]=0.943
            self.A0[1,1]=0.943
            self.A0[2,2]=2.407
            self.A0[3,3]=1.951
            self.A0[4,4]=1.951
            self.A0[5,5]=2.407
            self.A1=np.zeros((6,6),dtype=np.complex128)
            self.A1[0,0]=-0.207
            self.A1[0,2]=0.343654-0.323148j
            self.A1[0,4]=0.343654+0.323148j
            self.A1[1,1]=-0.207
            self.A1[1,3]=0.343654-0.323148j
            self.A1[1,5]=0.343654+0.323148j
            self.A1[2,0]=0.343654-0.323148j
            self.A1[2,2]=0.1485+0.329j
            self.A1[2,4]=-0.1145
            self.A1[3,1]=0.343654-0.323148j
            self.A1[3,3]=0.1485+0.329j
            self.A1[3,5]=-0.1145
            self.A1[4,0]=0.343654+0.323148j
            self.A1[4,2]=-0.1145
            self.A1[4,4]=0.1485-0.329j
            self.A1[5,1]=0.343654+0.323148j
            self.A1[5,3]=-0.1145
            self.A1[5,5]=0.1485-0.329j
            self.A2=np.zeros((6,6),dtype=np.complex128)
            self.A2[0,0]=-0.207
            self.A2[0,2]=-0.451681+0.136039j
            self.A2[0,4]=-0.451681-0.136039j
            self.A2[1,1]=-0.207
            self.A2[1,3]=-0.451681+0.136039j
            self.A2[1,5]=-0.451681-0.136039j
            self.A2[2,0]=0.108027-0.459187j
            self.A2[2,2]=0.1485-0.329j
            self.A2[2,4]=0.05725-0.0991599j
            self.A2[3,1]=0.108027-0.459187j
            self.A2[3,3]=0.1485-0.329j
            self.A2[3,5]=0.05725-0.0991599j
            self.A2[4,0]=0.108027+0.459187j
            self.A2[4,2]=0.05725+0.0991599j
            self.A2[4,4]=0.1485+0.329j
            self.A2[5,1]=0.108027+0.459187j
            self.A2[5,3]=0.05725+0.0991599j
            self.A2[5,5]=0.1485+0.329j
            self.A3=np.zeros((6,6),dtype=np.complex128)
            self.A3[0,0]=-0.207
            self.A3[0,2]=-0.451681-0.136039j
            self.A3[0,4]=-0.451681+0.136039j
            self.A3[1,1]=-0.207
            self.A3[1,3]=-0.451681-0.136039j
            self.A3[1,5]=-0.451681+0.136039j
            self.A3[2,0]=0.108027+0.459187j
            self.A3[2,2]=0.1485+0.329j
            self.A3[2,4]=0.05725+0.0991599j
            self.A3[3,1]=0.108027+0.459187j
            self.A3[3,3]=0.1485+0.329j
            self.A3[3,5]=0.05725+0.0991599j
            self.A3[4,0]=0.108027-0.459187j
            self.A3[4,2]=0.05725-0.0991599j
            self.A3[4,4]=0.1485-0.329j
            self.A3[5,1]=0.108027-0.459187j
            self.A3[5,3]=0.05725-0.0991599j
            self.A3[5,5]=0.1485-0.329j
            self.A4 = np.transpose(np.conj(self.A1))
            self.A5 = np.transpose(np.conj(self.A2))
            self.A6 = np.transpose(np.conj(self.A3))
        else:
            raise ValueError('Material "',mat_name,'" database is not bulit.')