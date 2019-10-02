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
    def __parameters__(self, mat_name):
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
        elif mat_name == 'WSe2_M10':
            '''
            WSe2 10x10 const.
            '''
            self.acc = 3.125e-10                  # m. nearest neighbor distance
            self.a = self.acc*3**0.5                # m. carbon to carbon distance
            self.ax = 3*self.acc                    # m. unit cell width
            self.K_norm = 4/3*np.pi/self.acc      # m-1. normalized K vector
            self.SOI = 0.228                    # SOI parameter. spin-orbit gap = 2*SOI
            self.Bx = 0                         # x magnetic field (eV)
            self.By = 0                         # y magnetic field (eV)
            self.Bz = 0                         # z magnetic field (eV)
            self.Ez = 0                         # z electric field (eV)
            self.A0=np.zeros((10,10),dtype=np.complex128)
            self.A1=np.zeros((10,10),dtype=np.complex128)
            self.A2=np.zeros((10,10),dtype=np.complex128)
            self.A3=np.zeros((10,10),dtype=np.complex128)
            self.setATNR10()
        else:
            raise ValueError('Material "',mat_name,'" database is not bulit.')
    def setATNR10(self):
            ### hopping energy
            self.A0[0,0]=(0.943 +0j)+1. *self.Bz
            self.A0[0,1]=(0. +0j)+1. *self.By-(0. +1j) *self.Bx
            self.A0[0,7]=(0. +0j)+(1.22474 +0j) *self.SOI
            self.A0[1,0]=(0. +0j)+1. *self.By+(0. +1j) *self.Bx
            self.A0[1,1]=(0.943 +0j)-1 *self.Bz
            self.A0[1,8]=(0. +0j)+(1.22474 +0j) *self.SOI
            self.A0[2,2]=(2.179 +0j)+1. *self.SOI+1 *self.Bz
            self.A0[2,3]=(0. +0j)+1. *self.By-(0. +1j) *self.Bx
            self.A0[2,8]=(0. +0j)-(0. +3j) *self.Ez
            self.A0[3,2]=(0. +0j)+1. *self.By+(0. +1j) *self.Bx
            self.A0[3,3]=(2.179 +0j)-1. *self.SOI-1 *self.Bz
            self.A0[3,6]=(0. +0j)+1. *self.SOI
            self.A0[3,9]=(0. +0j)-(0. +3j) *self.Ez
            self.A0[4,4]=(2.179 +0j)-1. *self.SOI+1 *self.Bz
            self.A0[4,5]=(0. +0j)+1. *self.By-(0. +1j) *self.Bx
            self.A0[4,6]=(0. +0j)-(0. +3j) *self.Ez
            self.A0[4,9]=(0. +0j)+1. *self.SOI
            self.A0[5,4]=(0. +0j)+1. *self.By+(0. +1j) *self.Bx
            self.A0[5,5]=(2.179 +0j)+1. *self.SOI-1 *self.Bz
            self.A0[5,7]=(0. +0j)-(0. +3j) *self.Ez
            self.A0[6,3]=(0. +0j)+1. *self.SOI
            self.A0[6,4]=(0. +0j)+(0. +3j) *self.Ez
            self.A0[6,6]=(2.557 +0j)+0.5 *self.SOI+1 *self.Bz
            self.A0[6,7]=(0. +0j)+1. *self.By-(0. +1j) *self.Bx
            self.A0[7,0]=(0. +0j)+(1.22474 +0j) *self.SOI
            self.A0[7,5]=(0. +0j)+(0. +3j) *self.Ez
            self.A0[7,6]=(0. +0j)+1. *self.By+(0. +1j) *self.Bx
            self.A0[7,7]=(2.557 +0j)-0.5 *self.SOI-1 *self.Bz
            self.A0[8,1]=(0. +0j)+(1.22474 +0j) *self.SOI
            self.A0[8,2]=(0. +0j)+(0. +3j) *self.Ez
            self.A0[8,8]=(2.557 +0j)-0.5 *self.SOI+1. *self.Bz
            self.A0[8,9]=(0. +0j)+1. *self.By-(0. +1j) *self.Bx
            self.A0[9,3]=(0. +0j)+(0. +3j) *self.Ez
            self.A0[9,4]=(0. +0j)+1. *self.SOI
            self.A0[9,8]=(0. +0j)+1. *self.By+(0. +1j) *self.Bx
            self.A0[9,9]=(2.557 +0j)+0.5 *self.SOI-1. *self.Bz
            
            self.A1[0,0]=-0.207+0j
            self.A1[0,2]=0.343654 -0.323148j
            self.A1[0,4]=0.343654 +0.323148j
            self.A1[0,6]=(0. +0j)-(0.612372 +0.353553j) *self.Ez*4
            self.A1[0,8]=(0. +0j)+(0.612372 -0.353553j) *self.Ez*4
            self.A1[1,1]=-0.207+0j
            self.A1[1,3]=0.343654 -0.323148j
            self.A1[1,5]=0.343654 +0.323148j
            self.A1[1,7]=(0. +0j)-(0.612372 +0.353553j) *self.Ez*4
            self.A1[1,9]=(0. +0j)+(0.612372 -0.353553j) *self.Ez*4
            self.A1[2,0]=0.343654 -0.323148j
            self.A1[2,2]=0.1485 +0.329j
            self.A1[2,4]=-0.1145+0j
            self.A1[2,6]=(0. +0j)-(0. +1j) *self.Ez
            self.A1[2,8]=(0. +0j)+(0.866025 +0.5j) *self.Ez
            self.A1[3,1]=0.343654 -0.323148j
            self.A1[3,3]=0.1485 +0.329j
            self.A1[3,5]=-0.1145+0j
            self.A1[3,7]=(0. +0j)-(0. +1j) *self.Ez
            self.A1[3,9]=(0. +0j)+(0.866025 +0.5j) *self.Ez
            self.A1[4,0]=0.343654 +0.323148j
            self.A1[4,2]=-0.1145+0j
            self.A1[4,4]=0.1485 -0.329j
            self.A1[4,6]=(0. +0j)-(0.866025 -0.5j) *self.Ez
            self.A1[4,8]=(0. +0j)-(0. +1j) *self.Ez
            self.A1[5,1]=0.343654 +0.323148j
            self.A1[5,3]=-0.1145+0j
            self.A1[5,5]=0.1485 -0.329j
            self.A1[5,7]=(0. +0j)-(0.866025 -0.5j) *self.Ez
            self.A1[5,9]=(0. +0j)-(0. +1j) *self.Ez
            self.A1[6,0]=(0. +0j)+(0.612372 +0.353553j) *self.Ez*4
            self.A1[6,2]=(0. +0j)+(0. +1j) *self.Ez
            self.A1[6,4]=(0. +0j)+(0.866025 -0.5j) *self.Ez
            self.A1[6,6]=-0.0928167-0.19325j
            self.A1[6,8]=-0.0618833+0j
            self.A1[7,1]=(0. +0j)+(0.612372 +0.353553j) *self.Ez*4
            self.A1[7,3]=(0. +0j)+(0. +1j) *self.Ez
            self.A1[7,5]=(0. +0j)+(0.866025 -0.5j) *self.Ez
            self.A1[7,7]=-0.0928167-0.19325j
            self.A1[7,9]=-0.0618833+0j
            self.A1[8,0]=(0. +0j)-(0.612372 -0.353553j) *self.Ez*4
            self.A1[8,2]=(0. +0j)-(0.866025 +0.5j) *self.Ez
            self.A1[8,4]=(0. +0j)+(0. +1j) *self.Ez
            self.A1[8,6]=-0.0618833+0j
            self.A1[8,8]=-0.0928167+0.19325j
            self.A1[9,1]=(0. +0j)-(0.612372 -0.353553j) *self.Ez*4
            self.A1[9,3]=(0. +0j)-(0.866025 +0.5j) *self.Ez
            self.A1[9,5]=(0. +0j)+(0. +1j) *self.Ez
            self.A1[9,7]=-0.0618833+0j
            self.A1[9,9]=-0.0928167+0.19325j
            
            self.A2[0,0]=-0.207+0j
            self.A2[0,2]=-0.451681+0.136039j
            self.A2[0,4]=-0.451681-0.136039j
            self.A2[0,6]=(0. +0j)-(0.612372 +0.353553j) *self.Ez*4
            self.A2[0,8]=(0. +0j)+(0.612372 -0.353553j) *self.Ez*4
            self.A2[1,1]=-0.207+0j
            self.A2[1,3]=-0.451681+0.136039j
            self.A2[1,5]=-0.451681-0.136039j
            self.A2[1,7]=(0. +0j)-(0.612372 +0.353553j) *self.Ez*4
            self.A2[1,9]=(0. +0j)+(0.612372 -0.353553j) *self.Ez*4
            self.A2[2,0]=0.108027 -0.459187j
            self.A2[2,2]=0.1485 -0.329j
            self.A2[2,4]=0.05725 -0.0991599j
            self.A2[2,6]=(0. +0j)+(0.866025 +0.5j) *self.Ez
            self.A2[2,8]=(0. +0j)-(0.866025 -0.5j) *self.Ez
            self.A2[3,1]=0.108027 -0.459187j
            self.A2[3,3]=0.1485 -0.329j
            self.A2[3,5]=0.05725 -0.0991599j
            self.A2[3,7]=(0. +0j)+(0.866025 +0.5j) *self.Ez
            self.A2[3,9]=(0. +0j)-(0.866025 -0.5j) *self.Ez
            self.A2[4,0]=0.108027 +0.459187j
            self.A2[4,2]=0.05725 +0.0991599j
            self.A2[4,4]=0.1485 +0.329j
            self.A2[4,6]=(0. +0j)+(0.866025 +0.5j) *self.Ez
            self.A2[4,8]=(0. +0j)-(0.866025 -0.5j) *self.Ez
            self.A2[5,1]=0.108027 +0.459187j
            self.A2[5,3]=0.05725 +0.0991599j
            self.A2[5,5]=0.1485 +0.329j
            self.A2[5,7]=(0. +0j)+(0.866025 +0.5j) *self.Ez
            self.A2[5,9]=(0. +0j)-(0.866025 -0.5j) *self.Ez
            self.A2[6,0]=(0. +0j)-(0. +0.707107j) *self.Ez*4
            self.A2[6,2]=(0. +0j)+(0.866025 -0.5j) *self.Ez
            self.A2[6,4]=(0. +0j)-(0.866025 +0.5j) *self.Ez
            self.A2[6,6]=-0.0928167+0.19325j
            self.A2[6,8]=0.0309417 +0.0535925j
            self.A2[7,1]=(0. +0j)-(0. +0.707107j) *self.Ez*4
            self.A2[7,3]=(0. +0j)+(0.866025 -0.5j) *self.Ez
            self.A2[7,5]=(0. +0j)-(0.866025 +0.5j) *self.Ez
            self.A2[7,7]=-0.0928167+0.19325j
            self.A2[7,9]=0.0309417 +0.0535925j
            self.A2[8,0]=(0. +0j)-(0. +0.707107j) *self.Ez*4
            self.A2[8,2]=(0. +0j)+(0.866025 -0.5j) *self.Ez
            self.A2[8,4]=(0. +0j)-(0.866025 +0.5j) *self.Ez
            self.A2[8,6]=0.0309417 -0.0535925j
            self.A2[8,8]=-0.0928167-0.19325j
            self.A2[9,1]=(0. +0j)-(0. +0.707107j) *self.Ez*4
            self.A2[9,3]=(0. +0j)+(0.866025 -0.5j) *self.Ez
            self.A2[9,5]=(0. +0j)-(0.866025 +0.5j) *self.Ez
            self.A2[9,7]=0.0309417 -0.0535925j
            self.A2[9,9]=-0.0928167-0.19325j
            
            self.A3[0,0]=-0.207+0j
            self.A3[0,2]=-0.451681-0.136039j
            self.A3[0,4]=-0.451681+0.136039j
            self.A3[0,6]=(0. +0j)+(0.612372 -0.353553j) *self.Ez*4
            self.A3[0,8]=(0. +0j)-(0.612372 +0.353553j) *self.Ez*4
            self.A3[1,1]=-0.207+0j
            self.A3[1,3]=-0.451681-0.136039j
            self.A3[1,5]=-0.451681+0.136039j
            self.A3[1,7]=(0. +0j)+(0.612372 -0.353553j) *self.Ez*4
            self.A3[1,9]=(0. +0j)-(0.612372 +0.353553j) *self.Ez*4
            self.A3[2,0]=0.108027 +0.459187j
            self.A3[2,2]=0.1485 +0.329j
            self.A3[2,4]=0.05725 +0.0991599j
            self.A3[2,6]=(0. +0j)-(0.866025 -0.5j) *self.Ez
            self.A3[2,8]=(0. +0j)+(0.866025 +0.5j) *self.Ez
            self.A3[3,1]=0.108027 +0.459187j
            self.A3[3,3]=0.1485 +0.329j
            self.A3[3,5]=0.05725 +0.0991599j
            self.A3[3,7]=(0. +0j)-(0.866025 -0.5j) *self.Ez
            self.A3[3,9]=(0. +0j)+(0.866025 +0.5j) *self.Ez
            self.A3[4,0]=0.108027 -0.459187j
            self.A3[4,2]=0.05725 -0.0991599j
            self.A3[4,4]=0.1485 -0.329j
            self.A3[4,6]=(0. +0j)-(0.866025 -0.5j) *self.Ez
            self.A3[4,8]=(0. +0j)+(0.866025 +0.5j) *self.Ez
            self.A3[5,1]=0.108027 -0.459187j
            self.A3[5,3]=0.05725 -0.0991599j
            self.A3[5,5]=0.1485 -0.329j
            self.A3[5,7]=(0. +0j)-(0.866025 -0.5j) *self.Ez
            self.A3[5,9]=(0. +0j)+(0.866025 +0.5j) *self.Ez
            self.A3[6,0]=(0. +0j)-(0. +0.707107j) *self.Ez*4
            self.A3[6,2]=(0. +0j)-(0.866025 +0.5j) *self.Ez
            self.A3[6,4]=(0. +0j)+(0.866025 -0.5j) *self.Ez
            self.A3[6,6]=-0.0928167-0.19325j
            self.A3[6,8]=0.0309417 -0.0535925j
            self.A3[7,1]=(0. +0j)-(0. +0.707107j) *self.Ez*4
            self.A3[7,3]=(0. +0j)-(0.866025 +0.5j) *self.Ez
            self.A3[7,5]=(0. +0j)+(0.866025 -0.5j) *self.Ez
            self.A3[7,7]=-0.0928167-0.19325j
            self.A3[7,9]=0.0309417 -0.0535925j
            self.A3[8,0]=(0. +0j)-(0. +0.707107j) *self.Ez*4
            self.A3[8,2]=(0. +0j)-(0.866025 +0.5j) *self.Ez
            self.A3[8,4]=(0. +0j)+(0.866025 -0.5j) *self.Ez
            self.A3[8,6]=0.0309417 +0.0535925j
            self.A3[8,8]=-0.0928167+0.19325j
            self.A3[9,1]=(0. +0j)-(0. +0.707107j) *self.Ez*4
            self.A3[9,3]=(0. +0j)-(0.866025 +0.5j) *self.Ez
            self.A3[9,5]=(0. +0j)+(0.866025 -0.5j) *self.Ez
            self.A3[9,7]=0.0309417 +0.0535925j
            self.A3[9,9]=-0.0928167+0.19325j
            self.A4 = np.transpose(np.conj(self.A1))
            self.A5 = np.transpose(np.conj(self.A2))
            self.A6 = np.transpose(np.conj(self.A3))