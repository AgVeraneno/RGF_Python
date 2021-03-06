import copy, os
import numpy as np

class test():
    '''
    Unit cell object contain essential information of a unit cell
    H: Hamiltonian of the unit cell containing on site energy & interhopping.
    P_plus: forward inter unit cell hopping matrix
    P_minus: backward inter unit cell hopping matrix
    SU size: number of orbits used for hopping
    SU count: number of atoms contained in sub unit cell
    Any new material should contain these 5 parameters correctly
    '''
    def __init__(self, setup, job):
        self.SU_size = 1                        # sub unit cell size (number of hopping and spin)
        self.SU_count = 1                       # atom number for each sub unit cell
        '''
        Auto generate parameters
        '''
        self.mat = setup['material']            # unit cell material
        self.mesh = int(setup['kx_mesh'])       # band structure mesh
        self.ax = self.mat.ax                   # unit length
        self.__initialize__(setup, job)
        self.__gen_Hamiltonian__()
    def __initialize__(self, setup, job):
        '''
        matrix definition
        '''
        ## ribbon size
        self.W = job['width']
        self.L = max(job['length'])
        ## lattice type
        self.m_size = sum(self.W)
        ## Hamiltonian
        empty_matrix = np.zeros((self.m_size,self.m_size), dtype=np.complex128)
        self.H = copy.deepcopy(empty_matrix)
        self.Pf = copy.deepcopy(empty_matrix)
        self.Pb = copy.deepcopy(empty_matrix)
        '''
        energy definition
        '''
        self.gap = job['gap']
        self.Vtop = job['Vtop']
        self.Vbot = job['Vbot']
    def __gen_Hamiltonian__(self):
        self.__component__()
        self.__off_diagonal__()
        self.__on_site_energy__()
    def __on_site_energy__(self):
        W_sum = sum(self.W)
        ## Gap assign
        gap = np.eye(self.m_size, dtype=np.complex128)*1000
        w_shift = 0
        for w_idx, W in enumerate(self.W):
            for i in range(W):
                gap[w_shift+i,w_shift+i] = self.gap[w_idx]
            else:
                w_shift += W
        ## Voltage assign
        volt = np.eye(self.m_size, dtype=np.complex128)*1000
        w_shift = 0
        for w_idx, W in enumerate(self.W):
            Vt = self.Vtop[w_idx]
            Vb = self.Vbot[w_idx]
            if W > 1: dV = (Vt-Vb)/(W-1)
            else: dV = 0
            for i in range(W):
                volt[w_shift+i,w_shift+i] = Vb + i*dV
            else:
                w_shift += W
        ## combine with Hamiltonian
        self.H += gap
        self.H += volt
    def __off_diagonal__(self):
        W = sum(self.W)
        z_mat = np.zeros((W,W), dtype=np.complex128)
        H = self.__on_chain__
        self.H = H + np.transpose(np.conj(H))
        Pb = self.__inter_chainP__
        self.Pb = np.block(Pb)
        self.Pf = np.transpose(np.conj(self.Pb))
    def __component__(self):
        W = sum(self.W)
        self.__on_chain__ = np.zeros((W,W),dtype=np.complex128)
        self.__inter_chain__ = np.zeros((W,W),dtype=np.complex128)
        self.__inter_chainP__ = np.zeros((W,W),dtype=np.complex128)
        for i in range(W):
            # build on chain matrix
            if i < W-1: self.__on_chain__[i,i+1] = -self.mat.r0
            # build inter chain matrix (same layer)
            self.__inter_chainP__[i,i] = -self.mat.r0
class Square():
    '''
    Unit cell object contain essential information of a unit cell
    H: Hamiltonian of the unit cell containing on site energy & interhopping.
    P_plus: forward inter unit cell hopping matrix
    P_minus: backward inter unit cell hopping matrix
    SU size: number of orbits used for hopping
    SU count: number of atoms contained in sub unit cell
    Any new material should contain these 5 parameters correctly
    '''
    def __init__(self, setup, region):
        self.SU_size = 1                        # sub unit cell size (number of hopping and spin)
        self.SU_count = 2                       # atom number for each sub unit cell
        '''
        Auto generate parameters
        '''
        self.mat = setup['Material']            # unit cell material
        self.region = region                          # current job setting
        self.mesh = int(setup['mesh'])       # band structure mesh
        self.ax = self.mat.ax                   # unit length
        self.__initialize__(setup, region)
        self.__gen_Hamiltonian__()
    def __initialize__(self, setup, region):
        '''
        matrix definition
        '''
        ## ribbon size
        self.W = [w*2 for w in region['Width']]
        self.L = region['Length']
        ## lattice type
        if setup['Lattice'] == 'M':
            self.m_size = 2*sum(self.W)
            self.lattice = 'MLG'
        elif setup['Lattice'] == 'B':
            self.m_size = 4*sum(self.W)
            self.lattice = 'BLG'
        else:
            raise ValueError('Unresolved lattice:', setup['Lattice'])
        ## Hamiltonian
        empty_matrix = np.zeros((self.m_size,self.m_size), dtype=np.complex128)
        self.H = copy.deepcopy(empty_matrix)
        self.Pf = copy.deepcopy(empty_matrix)
        self.Pb = copy.deepcopy(empty_matrix)
        '''
        energy definition
        '''
        self.gap = region['gap']
        self.Vtop = []
        self.Vbot = []
        dV = region['Vtop']-region['Vbot']
        for Vdrop in region['Vdrop']:
            if Vdrop == 'x':
                self.Vtop.append(region['Vbot'])
                self.Vbot.append(region['Vbot'])
            elif Vdrop == 'o':
                self.Vtop.append(region['Vtop'])
                self.Vbot.append(region['Vbot'])
        self.Bz = []
        for i, w in enumerate(region['Width']):
            for wi in range(2*w):
                self.Bz.append(region['B']['z'][i])
        '''
        job default
        '''
        if self.region['E_idx'][0] == None: self.region['E_idx'] = range(0,self.m_size,1)
        if self.region['S_idx'][0] == None: self.region['S_idx'] = range(0,int(setup['mesh']),1)
    def __gen_Hamiltonian__(self):
        self.__component__()
        self.__off_diagonal__()
        self.__on_site_energy__()
    def __on_site_energy__(self):
        W_sum = sum(self.W)
        ## Gap assign
        gap = np.eye(self.m_size, dtype=np.complex128)*1000
        w_shift = 0
        for w_idx, W in enumerate(self.W):
            for i in range(W):
                gap[w_shift+i,w_shift+i] = self.gap[w_idx]
                gap[w_shift+W_sum+i,w_shift+W_sum+i] = -self.gap[w_idx]
            else:
                w_shift += W
        ## Voltage assign
        volt = np.eye(self.m_size, dtype=np.complex128)*0
        w_shift = 0
        for w_idx, W in enumerate(self.W):
            Vt = self.Vtop[w_idx]
            Vb = self.Vbot[w_idx]
            if W > 1: dV = (Vt-Vb)/(3*W/2-2)
            else: dV = 0
            for i in range(W):
                volt[w_shift+i,w_shift+i] = Vb + i*dV + dV*int(i/2)
                volt[w_shift+W+i,w_shift+W+i] = Vb + i*dV + dV*int(i/2)
            else:
                w_shift += W
        ## combine with Hamiltonian
        if self.lattice == 'MLG':
            self.H = self.H[:self.m_size,:self.m_size]
            self.Pf = self.Pf[:self.m_size,:self.m_size]
            self.Pb = self.Pb[:self.m_size,:self.m_size]
        self.H += gap
        self.H += volt
        self.V = np.real(volt)
    def __off_diagonal__(self):
        W = sum(self.W)
        z_mat = np.zeros((W,W), dtype=np.complex128)
        H = [[self.__on_chain__, self.__inter_chain__],
             [z_mat, self.__on_chain__]]
        H = np.block(H)
        self.H = H + np.transpose(np.conj(H))
        Pf = [[z_mat,z_mat],
              [self.__on_chainP__,z_mat]]
        self.Pb = np.block(Pf)
        self.Pf = np.transpose(np.conj(self.Pb))
    def __component__(self):
        """
        1)
        matrix element definition
        |-----------------|
        |   C1   |  C1c1  |
        |-----------------|
        |   c1C1 |  c1    |
        |-----------------|
        2)
        ==============================================
        A5  O
            |
            |
        B4  X 
            |
            |
        A3  O
            |
            |
        B2  X
            |
            |
        A1  O
        ==============================================     
        """
        W = sum(self.W)
        self.__on_chain__ = np.zeros((W,W),dtype=np.complex128)
        self.__inter_chain__ = np.zeros((W,W),dtype=np.complex128)
        self.__on_chainP__ = np.zeros((W,W),dtype=np.complex128)
        for i in range(W):
            # build on chain matrix
            if i < W-1:
                self.__on_chain__[i,i+1] = -0.28
            self.__inter_chain__[i,i] = -0.28
            self.__on_chainP__[i,i] = -0.28
class AGNR():
    '''
    Unit cell object contain essential information of a unit cell
    H: Hamiltonian of the unit cell containing on site energy & interhopping.
    P_plus: forward inter unit cell hopping matrix
    P_minus: backward inter unit cell hopping matrix
    SU size: number of orbits used for hopping
    SU count: number of atoms contained in sub unit cell
    Any new material should contain these 5 parameters correctly
    '''
    def __init__(self, setup, region):
        self.SU_size = 1                        # sub unit cell size (number of hopping and spin)
        self.SU_count = 2                       # atom number for each sub unit cell
        '''
        Auto generate parameters
        '''
        self.mat = setup['Material']            # unit cell material
        self.mesh = int(setup['mesh'])          # band structure mesh
        self.region = region                    # current job setting
        self.ax = self.mat.ax                   # unit length
        self.__initialize__(setup, region)
        self.__gen_Hamiltonian__()
    def __initialize__(self, setup, region):
        '''
        matrix definition
        '''
        ## ribbon size
        self.W = [w*2 for w in region['Width']]
        self.L = region['Length']
        ## lattice type
        if setup['Lattice'] == 'M':
            self.m_size = 2*sum(self.W)
            self.__m_size__ = self.m_size
            self.lattice = 'MLG'
        elif setup['Lattice'] == 'B':
            self.m_size = 4*sum(self.W)
            self.__m_size__ = self.m_size
            self.lattice = 'BLG'
        else:
            raise ValueError('Unresolved lattice:', setup['Lattice'])
        ## Hamiltonian
        empty_matrix = np.zeros((self.m_size,self.m_size), dtype=np.complex128)
        self.H = copy.deepcopy(empty_matrix)
        self.Pf = copy.deepcopy(empty_matrix)
        self.Pb = copy.deepcopy(empty_matrix)
        '''
        energy definition
        '''
        self.gap = region['gap']
        self.Vtop = []
        self.Vbot = []
        dV = region['Vtop']-region['Vbot']
        for Vdrop in region['Vdrop']:
            if Vdrop == 'x':
                self.Vtop.append(region['Vbot'])
                self.Vbot.append(region['Vbot'])
            elif Vdrop == 'o':
                self.Vtop.append(region['Vtop'])
                self.Vbot.append(region['Vbot'])
        self.Bz = []
        for i, w in enumerate(region['Width']):
            for wi in range(2*w):
                self.Bz.append(region['B']['z'][i])
        '''
        job default
        '''
        if self.region['E_idx'][0] == None: self.region['E_idx'] = range(0,self.m_size,1)
        if self.region['S_idx'][0] == None: self.region['S_idx'] = range(0,int(setup['mesh']),1)
    def __gen_Hamiltonian__(self):
        self.__component__()
        self.__off_diagonal__()
        self.__on_site_energy__()
    def __on_site_energy__(self):
        W_sum = sum(self.W)
        ## Gap assign
        gap = np.eye(self.m_size, dtype=np.complex128)*1000
        w_shift = 0
        if self.lattice == 'MLG':
            for w_idx, W in enumerate(self.W):
                for i in range(W):
                    gap[w_shift+i,w_shift+i] = self.gap[w_idx] * (-1)**(i%2)
                    gap[W_sum+w_shift+i,W_sum+w_shift+i] = self.gap[w_idx] * (-1)**(1-i%2)
                else:
                    w_shift += W
        elif self.lattice == 'BLG':
            for w_idx, W in enumerate(self.W):
                for i in range(W):
                    gap[w_shift+i,w_shift+i] = self.gap[w_idx]
                    gap[W_sum+w_shift+i,W_sum+w_shift+i] = self.gap[w_idx]
                    gap[2*W_sum+w_shift+i,2*W_sum+w_shift+i] = -self.gap[w_idx]
                    gap[3*W_sum+w_shift+i,3*W_sum+w_shift+i] = -self.gap[w_idx]
                else:
                    w_shift += W
        ## Voltage assign
        volt = np.eye(self.m_size, dtype=np.complex128)*1000
        w_shift = 0
        if self.lattice == 'MLG':
            for w_idx, W in enumerate(self.W):
                Vt = self.Vtop[w_idx]
                Vb = self.Vbot[w_idx]
                if W > 1: dV = (Vt-Vb)/(W-1)
                else: dV = 0
                for i in range(W):
                    volt[w_shift+i,w_shift+i] = Vb + i*dV
                    volt[W_sum+w_shift+i,W_sum+w_shift+i] = Vb + i*dV
                else:
                    w_shift += W
        elif self.lattice == 'BLG':
            for w_idx, W in enumerate(self.W):
                Vt = self.Vtop[w_idx]
                Vb = self.Vbot[w_idx]
                if W > 1: dV = (Vt-Vb)/(W-1)
                else: dV = 0
                for i in range(W):
                    volt[w_shift+i,w_shift+i] = Vb + i*dV
                    volt[W_sum+w_shift+i,W_sum+w_shift+i] = Vb + i*dV
                    volt[2*W_sum+w_shift+i,2*W_sum+w_shift+i] = Vb + i*dV
                    volt[3*W_sum+w_shift+i,3*W_sum+w_shift+i] = Vb + i*dV
                else:
                    w_shift += W
        ## combine with Hamiltonian
        if self.lattice == 'MLG':
            self.H = self.H[:self.m_size,:self.m_size]
            self.Pf = self.Pf[:self.m_size,:self.m_size]
            self.Pb = self.Pb[:self.m_size,:self.m_size]
            self.uH = self.uH[:self.m_size,:self.m_size]
            self.uPf = self.uPf[:self.m_size,:self.m_size]
            self.uPb = self.uPb[:self.m_size,:self.m_size]
            self.uH0 = self.uH0[:self.m_size,:self.m_size]
            self.uPf0 = self.uPf0[:self.m_size,:self.m_size]
            self.uPb0 = self.uPb0[:self.m_size,:self.m_size]
        self.H += gap
        self.H += volt
        self.V = np.real(volt)
    def __off_diagonal__(self):
        W = sum(self.W)
        z_mat = np.zeros((W,W), dtype=np.complex128)
        H = [[self.__on_chain__, self.__inter_chain__, self.__C1c1__    , z_mat               ],
             [z_mat            , self.__on_chain__   , self.__C2c1__    , self.__C2c2__       ],
             [z_mat            , z_mat               , self.__on_chain__, self.__inter_chain__],
             [z_mat            , z_mat               , z_mat            , self.__on_chain__   ]]
        H = np.block(H)
        self.H = H + np.transpose(np.conj(H))
        Pb = [[z_mat            , self.__inter_chainP__, z_mat            , self.__C1c2__        ],
              [z_mat            , z_mat                , z_mat            , z_mat                ],
              [z_mat            , z_mat                , z_mat            , self.__inter_chainP__],
              [z_mat            , z_mat                , z_mat            , z_mat                ]]
        self.Pb = np.block(Pb)
        self.Pf = np.transpose(np.conj(self.Pb))
        '''
        Magnetic Hamiltonian
        '''
        uH = [[self.__Mu_on_chainA__, self.__Mu_inter_chain__, self.__C1c1__    , z_mat               ],
             [z_mat                 , self.__Mu_on_chainB__   , self.__C2c1__    , self.__C2c2__       ],
             [z_mat                 , z_mat               , self.__Mu_on_chainA__, self.__Mu_inter_chain__],
             [z_mat                 , z_mat               , z_mat            , self.__Mu_on_chainB__   ]]
        uH = np.block(uH)
        self.uH = uH + np.transpose(np.conj(uH))
        uPf = [[z_mat            , self.__Mu_inter_chainP__, z_mat            , self.__C1c2__        ],
              [z_mat            , z_mat                , z_mat            , z_mat                ],
              [z_mat            , z_mat                , z_mat            , self.__Mu_inter_chainP__],
              [z_mat            , z_mat                , z_mat            , z_mat                ]]
        self.uPf = np.block(uPf)
        self.uPb = np.transpose(np.conj(self.uPf))
        uH0 = [[self.__Mu0_on_chainA__, self.__Mu0_inter_chain__, self.__C1c1__    , z_mat               ],
             [z_mat                 , self.__Mu0_on_chainB__   , self.__C2c1__    , self.__C2c2__       ],
             [z_mat                 , z_mat               , self.__Mu0_on_chainA__, self.__Mu0_inter_chain__],
             [z_mat                 , z_mat               , z_mat            , self.__Mu0_on_chainB__   ]]
        uH0 = np.block(uH0)
        self.uH0 = uH0 + np.transpose(np.conj(uH0))
        uPf0 = [[z_mat            , self.__Mu_inter_chainP__, z_mat            , self.__C1c2__        ],
              [z_mat            , z_mat                , z_mat            , z_mat                ],
              [z_mat            , z_mat                , z_mat            , self.__Mu_inter_chainP__],
              [z_mat            , z_mat                , z_mat            , z_mat                ]]
        self.uPf0 = np.block(uPf0)
        self.uPb0 = np.transpose(np.conj(self.uPf0))
    def genPositionOperator(self, W):
        self.__Xop__ = np.zeros((2*W,2*W),dtype=np.complex128)
        self.__Yop__ = np.zeros((2*W,2*W),dtype=np.complex128)
        self.Y = np.zeros((2*W,2*W),dtype=np.complex128)
        for i in range(W):
            # build position matrix
            if i > 0:
                self.__Yop__[i,i] = self.__Yop__[i-1,i-1] + self.mat.a/2
                self.__Yop__[i+W,i+W] = self.__Yop__[i-1,i-1] + self.mat.a/2
                self.Y[i,i] = self.Y[i-1,i-1] + self.mat.a/2
                self.Y[i+W,i+W] = self.Y[i-1,i-1] + self.mat.a/2
                self.__Xop__[i,i] = -self.mat.acc/2*(i%2)
                self.__Xop__[i+W,i+W] = self.mat.acc+self.mat.acc/2*(i%2)
                for j in range(i):
                    self.__Yop__[i,j] = (self.__Yop__[i,i] + self.__Yop__[j,j])/2
                    self.__Yop__[i+W,j] = (self.__Yop__[i,i] + self.__Yop__[j,j])/2
                    self.__Yop__[j,i] = (self.__Yop__[i,i] + self.__Yop__[j,j])/2
                    self.__Yop__[j,i+W] = (self.__Yop__[i,i] + self.__Yop__[j,j])/2
                    self.__Xop__[i,j] = (self.__Xop__[i,i] - self.__Xop__[j,j])
                    self.__Xop__[i+W,j] = (self.__Xop__[i+W,i+W] - self.__Xop__[j,j])
                    self.__Xop__[j,i] = (self.__Xop__[j,j] - self.__Xop__[i,i])
                    self.__Xop__[j,i+W] = (self.__Xop__[j,j] - self.__Xop__[i+W,i+W])
    def __component__(self):
        """
        1)
        matrix element definition
        |-----------------------------------|
        |   C1   |  C1C2  |   C1c1 |   C1c2 |
        |-----------------------------------|
        |   C2C1 |  C2    |   C2c1 |  C2c2  |
        |-----------------------------------|
        |   c1C1 |  c1C2  |   c1   |  c1c2  |
        |-----------------------------------|
        |   c2C1 |  c2C2  |   c2c1 |  c2    |
        |-----------------------------------|
        2)
        ==============================================
            A3   B3,a3   b3
            X------0======O       
           /      > \      <
          /      >   \      <
         B2    b2    A2      a2
         X     O     X       O
          \     <   /       >
           \     < /       >
           X------0======O
           A1   B1,a1   b1
           
           ^    ^    ^   ^
           C1   c1  C2   c2
        ==============================================     
        """
        W = sum(self.W)
        self.__on_chain__ = np.zeros((W,W),dtype=np.complex128)
        self.__inter_chain__ = np.zeros((W,W),dtype=np.complex128)
        self.__inter_chainP__ = np.zeros((W,W),dtype=np.complex128)
        self.__C1c1__ = np.zeros((W,W),dtype=np.complex128)
        self.__C1c2__ = np.zeros((W,W),dtype=np.complex128)
        self.__C2c1__ = np.zeros((W,W),dtype=np.complex128)
        self.__C2c2__ = np.zeros((W,W),dtype=np.complex128)
        self.__Mu_on_chainA__ = np.zeros((W,W),dtype=np.complex128)
        self.__Mu_inter_chain__ = np.zeros((W,W),dtype=np.complex128)
        self.__Mu_inter_chainP__ = np.zeros((W,W),dtype=np.complex128)
        self.__Mu0_on_chainA__ = np.zeros((W,W),dtype=np.complex128)
        self.__Mu0_inter_chain__ = np.zeros((W,W),dtype=np.complex128)
        self.__Mu0_inter_chainP__ = np.zeros((W,W),dtype=np.complex128)
        self.__Mu_on_chainB__ = np.zeros((W,W),dtype=np.complex128)
        self.__Mu0_on_chainB__ = np.zeros((W,W),dtype=np.complex128)
        self.genPositionOperator(W)
        mu_const = 1j*self.mat.q/self.mat.h_bar
        for i in range(W):
            # build on chain matrix
            if i < W-1:
                self.__on_chain__[i,i+1] = -self.mat.r0
                self.__Mu_on_chainA__[i,i+1] = mu_const*self.mat.r0*self.mat.q*self.__Xop__[i,i+1]*self.__Yop__[i,i+1]*\
                                                np.exp(mu_const*self.Bz[i]*self.__Xop__[i,i+1]*self.__Yop__[i,i+1])
                self.__Mu0_on_chainA__[i,i+1] = mu_const*self.mat.r0*self.mat.q*self.__Xop__[i,i+1]*\
                                                np.exp(mu_const*self.Bz[i]*self.__Xop__[i,i+1])
                self.__Mu_on_chainB__[i,i+1] = mu_const*self.mat.r0*self.mat.q*self.__Xop__[i+W,i+W+1]*self.__Yop__[i+W,i+W+1]*\
                                                np.exp(mu_const*self.Bz[i]*self.__Xop__[i+W,i+W+1]*self.__Yop__[i+W,i+W+1])
                self.__Mu0_on_chainB__[i,i+1] = mu_const*self.mat.r0*self.mat.q*self.__Xop__[i+W,i+W+1]*\
                                                np.exp(mu_const*self.Bz[i]*self.__Xop__[i+W,i+W+1])
            # build inter chain matrix (same layer)
            self.__inter_chain__[i,i] = -self.mat.r0 * (1-i%2)
            self.__inter_chainP__[i,i] = -self.mat.r0 * (i%2)
            self.__Mu0_inter_chain__[i,i] = mu_const*self.mat.r0*self.mat.q*self.__Xop__[i,i+W]*\
                                            np.exp(mu_const*self.Bz[i]*self.__Xop__[i,i+W]) * (1-i%2)
            self.__Mu0_inter_chainP__[i,i] = mu_const*self.mat.r0*self.mat.q*self.__Xop__[i,i+W]*\
                                            np.exp(mu_const*self.Bz[i]*self.__Xop__[i,i+W]) * (i%2)
            self.__Mu_inter_chainP__[i,i] = mu_const*self.mat.r0*self.mat.q*self.mat.acc*self.__Yop__[i,i+W]*\
                                            np.exp(mu_const*self.Bz[i]*self.mat.acc*self.__Yop__[i,i+W]) * (i%2)
            # build inter chain matrix (C1 to c1)
            if i < W-1: self.__C1c1__[i,i+1] = -self.mat.r3 * (1-i%2)
            if i > 0: self.__C1c1__[i,i-1] = -self.mat.r3 * (1-i%2)
            # build inter chain matrix (C1 to c2)
            self.__C1c2__[i,i] = -self.mat.r3 * (1-i%2) - self.mat.r1 * (i%2)
            # build inter chain matrix (C2 to c1)
            self.__C2c1__[i,i] = -self.mat.r1 * (1-i%2) - self.mat.r3 * (i%2)
            # build inter chain matrix (C2 to c2)
            if i < W-1: self.__C2c2__[i,i+1] = -self.mat.r3 * (i%2)
            if i > 0: self.__C2c2__[i,i-1] = -self.mat.r3 * (i%2)
class ZGNR():
    '''
    Unit cell object contain essential information of a unit cell
    H: Hamiltonian of the unit cell containing on site energy & interhopping.
    P_plus: forward inter unit cell hopping matrix
    P_minus: backward inter unit cell hopping matrix
    SU size: number of orbits used for hopping
    SU count: number of atoms contained in sub unit cell
    Any new material should contain these 5 parameters correctly
    '''
    def __init__(self, setup, region):
        self.SU_size = 1                        # sub unit cell size (number of hopping and spin)
        self.atomPerWidth = 2                   # atom number for each width
        '''
        Auto generate parameters
        '''
        self.mat = setup['Material']            # unit cell material
        self.region = region                          # current job setting
        self.mesh = int(setup['mesh'])       # band structure mesh
        self.__initialize__(setup, region)
        self.__gen_Hamiltonian__()
    def __initialize__(self, setup, region):
        '''
        matrix definition
        '''
        ## ribbon size
        self.__W__ = [self.atomPerWidth*w for w in region['Width']]         # each width contain 2 atoms
        self.__W__.reverse() 
        self.__Wtot__ = sum(self.__W__)
        self.L = region['Length']
        ## lattice type
        self.__lattice__ = setup['Lattice']
        if setup['Lattice'] == 'M': self.__m_size__ = self.__Wtot__
        elif setup['Lattice'] == 'B': self.__m_size__ = 2*self.__Wtot__
        else: raise ValueError('Unresolved lattice:', setup['Lattice'])
        ## Hamiltonian
        empty_matrix = np.zeros((self.__m_size__,self.__m_size__), dtype=np.complex128)
        self.H = copy.deepcopy(empty_matrix)
        self.Pf = copy.deepcopy(empty_matrix)
        self.Pb = copy.deepcopy(empty_matrix)
        '''
        energy definition
        '''
        self.__gap__ = region['gap']
        self.Vtop = []
        self.Vbot = []
        dV = region['Vtop']-region['Vbot']
        for Vdrop in region['Vdrop']:
            if Vdrop == 'x':
                self.Vtop.append(region['Vbot'])
                self.Vbot.append(region['Vbot'])
            elif Vdrop == 'o':
                self.Vtop.append(region['Vtop'])
                self.Vbot.append(region['Vbot'])
        self.Bz = []
        for i, w in enumerate(region['Width']):
            for wi in range(2*w):
                self.Bz.append(region['B']['z'][i])
        '''
        job default
        '''
        if self.region['E_idx'][0] == None: self.region['E_idx'] = range(0,self.__m_size__,1)
        if self.region['S_idx'][0] == None: self.region['S_idx'] = range(0,self.mesh,1)
    def __gen_Hamiltonian__(self):
        self.__component__()
        self.__off_diagonal__()
        self.__on_site_energy__()
    def __on_site_energy__(self):
        W_sum = sum(self.__W__)
        ## Gap assign
        gap = np.eye(self.__m_size__, dtype=np.complex128)*1e7
        w_shift = 0
        if self.__lattice__ == 'M':
            for w_idx, W in enumerate(self.__W__):
                for i in range(W): gap[w_shift+i,w_shift+i] = self.__gap__[w_idx] * (-1)**(i%2)
                else: w_shift += W
        elif self.__lattice__ == 'B':
            for w_idx, W in enumerate(self.__W__):
                for i in range(W):
                    gap[w_shift+i,w_shift+i] = self.__gap__[w_idx]
                    gap[W_sum+w_shift+i,W_sum+w_shift+i] = self.__gap__[w_idx]
                    w_shift += W
        ###
        self.delta = self.__gap__[0]
        ###
        ## Voltage assign
        volt = np.eye(self.__m_size__, dtype=np.complex128)*1e7
        w_shift = 0
        if self.__lattice__ == 'M':
            for w_idx, W in enumerate(self.__W__):
                Vt = self.Vtop[w_idx]
                Vb = self.Vbot[w_idx]
                if W > 1: dV = (Vt-Vb)/(3*W/2-2)
                else: dV = 0
                for i in range(W):
                    volt[w_shift+i,w_shift+i] = Vb + i*dV + dV*int(i/2)
                else:
                    w_shift += W
        elif self.__lattice__ == 'B':
            for w_idx, W in enumerate(self.__W__):
                Vt = self.Vtop[w_idx]
                Vb = self.Vbot[w_idx]
                if W > 1: dV = (Vt-Vb)/(W-1)
                else: dV = 0
                for i in range(W):
                    volt[w_shift+i,w_shift+i] = Vb+dV + i*dV + i*dV*(i%2)
                    volt[W_sum+w_shift+i,W_sum+w_shift+i] = Vb+dV + i*dV + i*dV*(i%2)
                else:
                    w_shift += W
        ## combine with Hamiltonian
        if self.__lattice__ == 'M':
            self.H = self.H[:self.__m_size__,:self.__m_size__]
            self.Pf = self.Pf[:self.__m_size__,:self.__m_size__]
            self.Pb = self.Pb[:self.__m_size__,:self.__m_size__]
            self.uH = self.uH[:self.__m_size__,:self.__m_size__]
            self.uPf = self.uPf[:self.__m_size__,:self.__m_size__]
            self.uPb = self.uPb[:self.__m_size__,:self.__m_size__]
            self.uH0 = self.uH0[:self.__m_size__,:self.__m_size__]
            self.uPf0 = self.uPf0[:self.__m_size__,:self.__m_size__]
            self.uPb0 = self.uPb0[:self.__m_size__,:self.__m_size__]
        self.H += gap
        #self.H += volt
        self.V = np.real(volt)
    def __off_diagonal__(self):
        W = sum(self.__W__)
        z_mat = np.zeros((W,W), dtype=np.complex128)
        H = [[self.__on_chain__, self.__C1c1__    ],
             [z_mat            , self.__on_chain__]]
        H = np.block(H)
        self.H = H + np.transpose(np.conj(H))
        Pf = [[self.__inter_chainP__, z_mat],
              [z_mat, self.__inter_chainP__]]
        self.Pf = np.block(Pf)
        self.Pb = np.transpose(np.conj(self.Pf))
        '''
        Magnetic Hamiltonian
        '''
        uH = [[self.__Mu_on_chain__, self.__C1c1__],
             [z_mat            , self.__Mu_on_chain__]]
        uH = np.block(uH)
        self.uH = uH + np.transpose(np.conj(uH))
        uPf = [[self.__Mu_on_chainP__, self.__C1c1P__    ],
              [z_mat             , self.__Mu_on_chainP__]]
        self.uPf = np.block(uPf)
        self.uPb = np.transpose(np.conj(self.uPf))
        uH0 = [[self.__Mu0_on_chain__, self.__C1c1__],
             [z_mat            , self.__Mu0_on_chain__]]
        uH0 = np.block(uH0)
        self.uH0 = uH0 + np.transpose(np.conj(uH0))
        uPf0 = [[self.__Mu0_on_chainP__, self.__C1c1P__    ],
              [z_mat             , self.__Mu0_on_chainP__]]
        self.uPf0 = np.block(uPf0)
        self.uPb0 = np.transpose(np.conj(self.uPf0))
    def genPositionOperator(self, W):
        self.__Xop__ = np.zeros((self.__m_size__,self.__m_size__))
        self.__Yop__ = np.zeros((self.__m_size__,self.__m_size__))
        self.Y = np.zeros((self.__m_size__,self.__m_size__))
        for i in range(W):
            # build position matrix
            if i > 0:
                self.__Yop__[i,i] = self.__Yop__[i-1,i-1] + self.mat.acc/2*(i%2) + self.mat.acc*((i+1)%2)
                self.Y[i,i] = self.__Yop__[i,i]
                if i%4 == 1 or i%4 == 2: self.__Xop__[i,i] = self.mat.a/2
                else: self.__Xop__[i,i] = 0
                for j in range(i):
                    self.__Yop__[i,j] = (self.__Yop__[i,i] + self.__Yop__[j,j])/2
                    self.__Yop__[j,i] = self.__Yop__[i,j]
                    self.__Xop__[i,j] = (self.__Xop__[i,i] - self.__Xop__[j,j])
                    self.__Xop__[j,i] = (self.__Xop__[j,j] - self.__Xop__[i,i])
    def __component__(self):
        """
        1)
        matrix element definition
        |-----------------|
        |   C1   |  C1c1  |
        |-----------------|
        |   c1C1 |  c1    |
        |-----------------|
        2)
        ==============================================
              B6 0 a5
                /|
              /  |
        A5  O    |
            |    X b4
            |     <
            |      <
            |       < 
        B4  O        X a3
               \       ^
                \      ^
              A3  O    ^
                  |    ^
                  |    ^
                  |    X b2
                  |   >
                  | >
              B2  0 a1
                /
              /
        A1  O      

           
           ^    ^ 
           C1   c1
        ==============================================     
        """
        W = sum(self.__W__)
        self.genPositionOperator(W)
        self.__on_chain__ = np.zeros((W,W),dtype=np.complex128)
        self.__inter_chainP__ = np.zeros((W,W),dtype=np.complex128)
        self.__C1c1__ = np.zeros((W,W),dtype=np.complex128)
        self.__C1c2__ = np.zeros((W,W),dtype=np.complex128)
        self.__C1c1P__ = np.zeros((W,W),dtype=np.complex128)
        self.__C2c1__ = np.zeros((W,W),dtype=np.complex128)
        self.__C2c2__ = np.zeros((W,W),dtype=np.complex128)
        self.__Mu_on_chain__ = np.zeros((W,W),dtype=np.complex128)
        self.__Mu_on_chainP__ = np.zeros((W,W),dtype=np.complex128)
        self.__Mu_C1c1__ = np.zeros((W,W),dtype=np.complex128)
        self.__Mu_C1c1P__ = np.zeros((W,W),dtype=np.complex128)
        self.__Mu0_on_chain__ = np.zeros((W,W),dtype=np.complex128)
        self.__Mu0_on_chainP__ = np.zeros((W,W),dtype=np.complex128)
        self.__Mu0_C1c1__ = np.zeros((W,W),dtype=np.complex128)
        self.__Mu0_C1c1P__ = np.zeros((W,W),dtype=np.complex128)
        mu_const = 1j*self.mat.q/self.mat.h_bar
        for i in range(W):
            # build on chain matrix
            if i < W-1:
                self.__on_chain__[i,i+1] = -self.mat.r0
                self.__Mu_on_chain__[i,i+1] = mu_const*self.mat.r0*self.mat.q*self.__Xop__[i,i+1]*self.__Yop__[i,i+1]*\
                                                np.exp(mu_const*self.Bz[i]*self.__Xop__[i,i+1]*self.__Yop__[i,i+1])
                self.__Mu0_on_chain__[i,i+1] = mu_const*self.mat.r0*self.mat.q*self.__Xop__[i,i+1]*\
                                                np.exp(mu_const*self.Bz[i]*self.__Xop__[i,i+1])
            # build inter chain matrix and hopping matrix
            if i < W-1 and i%4 == 0:
                self.__inter_chainP__[i,i+1] = -self.mat.r0
                dx = self.mat.a/2
                self.__Mu_on_chainP__[i,i+1] = mu_const*self.mat.r0*self.mat.q*dx*self.__Yop__[i,i+1]*\
                                                np.exp(mu_const*self.Bz[i]*dx*self.__Yop__[i,i+1])
                self.__Mu0_on_chainP__[i,i+1] = mu_const*self.mat.r0*self.mat.q*dx*\
                                                np.exp(mu_const*self.Bz[i]*dx)
            if i < W-1 and i%4 == 2:
                self.__inter_chainP__[i+1,i] = -self.mat.r0
                dx = self.mat.a/2
                self.__Mu_on_chainP__[i+1,i] = mu_const*self.mat.r0*self.mat.q*dx*self.__Yop__[i+1,i]*\
                                                np.exp(mu_const*self.Bz[i]*dx*self.__Yop__[i+1,i])
                self.__Mu0_on_chainP__[i+1,i] = mu_const*self.mat.r0*self.mat.q*dx*\
                                                np.exp(mu_const*self.Bz[i]*dx)
            # build inter chain matrix (C1 to c1)
            if i < W-1 and i%4 == 0: self.__C1c1P__[i,i+1] = -self.mat.r3
            if i > 0 and i%4 != 3: self.__C1c1__[i,i-1] = -self.mat.r3 * (1-i%2) - self.mat.r1 * (i%2)
            if i > 0 and i%4 == 3: self.__C1c1__[i,i-1] = -self.mat.r1
