import copy, os
import numpy as np

class AGNR():
    '''
    Unit cell object contain essential information of a unit cell
    H: Hamiltonian of the unit cell containing on site energy & interhopping.
    P_plus: forward inter unit cell hopping matrix
    P_minus: backward inter unit cell hopping matrix
    '''
    def __init__(self, setup, job):
        '''
        New material parameters
        SU size: number of orbits used for hopping
        SU count: number of atoms contained in sub unit cell
        '''
        self.SU_size = 1                        # sub unit cell size
        self.SU_count = 2
        '''
        Auto generate parameters
        '''
        self.__initialize__(setup, job)  # build unit cell geometry
        self.mesh = int(setup['kx_mesh'])       # band structure mesh
        self.ax = self.mat.ax                   # unit length
        self.__gen_Hamiltonian__()
    def __initialize__(self, setup, job):
        self.mat = setup['material']                # unit cell material
        #self.SU_type = setup['SU_type']             # sub unitcell type
        self.SU_type = 'separate'
        '''
        matrix definition
        '''
        ## ribbon size
        self.W = job['width']
        self.L = max(job['length'])
        ## lattice type
        if setup['lattice'] == 'MLG':
            self.m_size = 2*sum(self.W)
            self.gap_inv = 1
            self.L1_stop = [job['shift'][i]+W-1 for i,W in enumerate(np.cumsum(self.W))]
            self.L1_start = [self.L1_stop[i]-W+1 for i,W in enumerate(self.W)]
            self.L2_start = [-1 for i in range(len(self.W))]
            self.L2_stop = [-1 for i in range(len(self.W))]
        elif setup['lattice'] == 'BLG':
            self.m_size = 4*sum(self.W)
            self.gap_inv = 0
            self.L1_stop = [job['shift'][i]+W-1 for i,W in enumerate(np.cumsum(self.W))]
            self.L1_start = [self.L1_stop[i]-W+1 for i,W in enumerate(self.W)]
            self.L2_start = [self.L1_start[i]+int(self.m_size/2) for i in range(len(self.W))]
            self.L2_stop = [self.L1_stop[i]+int(self.m_size/2) for i in range(len(self.W))]
        else:
            raise ValueError('Unresolved lattice:', setup['lattice'])
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
        ## sub unit cell size
        SU = [int(W/2) for W in self.W]
        SU_add = [W%2 for W in self.W]
        W = 2*sum(self.W)
        if self.SU_type == 'separate':
            SU_sep = [SU[i] + SU_add[i] for i in range(len(self.W))]
            SU_ovl = SU
        elif self.SU_type == 'overlap':
            SU_sep = SU
            SU_ovl = [SU[i] + SU_add[i] for i in range(len(self.W))]
        else:
            raise ValueError('Unresolved type:',self.SU_type)
        SU_shift = sum(SU_sep)
        '''
        Gap Profile
        '''
        gap_profile = np.eye(self.m_size, dtype=np.complex128)*1000
        ## separate type sub unit
        shift = 0
        for i in range(len(self.W)):
            gap = self.gap[i]
            for j, sep in enumerate(range(SU_sep[i])):
                m = 2*(shift+sep)
                if self.gap_inv:        # MLG
                    gap_profile[m,m] = gap
                    gap_profile[m+1,m+1] = -gap
                else:       # BLG
                    gap_profile[m,m] = gap
                    gap_profile[m+1,m+1] = gap
                    gap_profile[m+W,m+W] = -gap
                    gap_profile[m+W+1,m+W+1] = -gap
            else:
                shift += SU_sep[i]
        ## overlap type sub unit
        shift = sum(SU_sep)
        for i in range(len(self.W)):
            gap = self.gap[i]
            for j, sep in enumerate(range(SU_ovl[i])):
                m = 2*(shift+sep)
                if self.gap_inv:        # MLG
                    gap_profile[m,m] = gap
                    gap_profile[m+1,m+1] = -gap
                else:       # BLG
                    gap_profile[m,m] = gap
                    gap_profile[m+1,m+1] = gap
                    gap_profile[m+W,m+W] = -gap
                    gap_profile[m+W+1,m+W+1] = -gap
            else:
                shift += SU_ovl[i]
        '''
        Voltage profile
        '''
        dv_profile = np.zeros((self.m_size,self.m_size), dtype=np.complex128)
        ## separate type sub unit
        shift = 0
        for i in range(len(self.W)):
            Vtop = self.Vtop[i]
            Vbot = self.Vbot[i]
            dV = (Vtop - Vbot)/(self.W[i]+1)
            for j, sep in enumerate(range(SU_sep[i])):
                m = 2*(shift+sep)
                if self.gap_inv:        # MLG
                    if self.SU_type == 'separate':
                        dv_profile[m,m] = Vbot+2*(j+1)*dV
                        dv_profile[m+1,m+1] = Vbot+2*(j+1)*dV
                    elif self.SU_type == 'overlap':
                        dv_profile[m,m] = Vbot+2*(j+1.5)*dV
                        dv_profile[m+1,m+1] = Vbot+2*(j+1.5)*dV
                    else:
                        raise ValueError('Unresolved type:',self.SU_type)
                else:       # BLG
                    if self.SU_type == 'separate':
                        dv_profile[m,m] = Vbot+2*(j+0.5)*dV
                        dv_profile[m+1,m+1] = Vbot+2*(j+0.5)*dV
                        dv_profile[m+W,m+W] = Vbot+2*(j+0.5)*dV
                        dv_profile[m+W+1,m+W+1] = Vbot+2*(j+0.5)*dV
                    elif self.SU_type == 'overlap':
                        dv_profile[m,m] = Vbot+2*(j+1)*dV
                        dv_profile[m+1,m+1] = Vbot+2*(j+1)*dV
                        dv_profile[m+W,m+W] = Vbot+2*(j+1)*dV
                        dv_profile[m+W+1,m+W+1] = Vbot+2*(j+1)*dV
                    else:
                        raise ValueError('Unresolved type:',self.SU_type)
            else:
                shift += SU_sep[i]
        ## overlap type sub unit
        shift = sum(SU_sep)
        for i in range(len(self.W)):
            Vtop = self.Vtop[i]
            Vbot = self.Vbot[i]
            dV = (Vtop - Vbot)/(self.W[i]+1)
            for j, sep in enumerate(range(SU_ovl[i])):
                m = 2*(shift+sep)
                if self.gap_inv:        # MLG
                    if self.SU_type == 'separate':
                        dv_profile[m,m] = Vbot+2*(j+1.5)*dV
                        dv_profile[m+1,m+1] = Vbot+2*(j+1.5)*dV
                    elif self.SU_type == 'overlap':
                        dv_profile[m,m] = Vbot+2*(j+1)*dV
                        dv_profile[m+1,m+1] = Vbot+2*(j+1)*dV
                    else:
                        raise ValueError('Unresolved type:',self.SU_type)
                else:       # BLG
                    if self.SU_type == 'separate':
                        dv_profile[m,m] = Vbot+2*(j+1)*dV
                        dv_profile[m+1,m+1] = Vbot+2*(j+1)*dV
                        dv_profile[m+W,m+W] = Vbot+2*(j+1)*dV
                        dv_profile[m+W+1,m+W+1] = Vbot+2*(j+1)*dV
                    elif self.SU_type == 'overlap':
                        dv_profile[m,m] = Vbot+2*(j+0.5)*dV
                        dv_profile[m+1,m+1] = Vbot+2*(j+0.5)*dV
                        dv_profile[m+W,m+W] = Vbot+2*(j+0.5)*dV
                        dv_profile[m+W+1,m+W+1] = Vbot+2*(j+0.5)*dV
                    else:
                        raise ValueError('Unresolved type:',self.SU_type)
            else:
                shift += SU_ovl[i]
        '''
        Combine
        '''
        self.H += gap_profile
        self.H += dv_profile
    def __off_diagonal__(self):
        empty_matrix = np.zeros((2*self.SU_size,2*self.SU_size), dtype=np.complex128)
        ## unit cell H
        H = []
        blockHAB = []
        blockHAC = []
        blockHAD = []
        blockHBB = []
        blockHBC = []
        blockHBD = []
        ## forward hopping matrix Pf
        Pf = []
        blockPAA = []
        blockPAC = []
        blockPBD = []
        ## sub unit cell size
        SU = int(sum(self.W)/2)
        SU_add = sum(self.W)%2
        if self.SU_type == 'separate':
            SU_sep = SU + SU_add
            SU_ovl = SU
        elif self.SU_type == 'overlap':
            SU_sep = SU
            SU_ovl = SU + SU_add
        else:
            raise ValueError('Unresolved type:',self.SU_type)
        ## AA and AC matrix
        for r in range(SU_sep):
            rowPAA = []
            rowHAC = []
            rowPAC = []
            for c in range(SU_sep):
                if r == c:
                    rowPAA.append(self.__PAAA__)
                    rowHAC.append(self.__MAC__)
                    rowPAC.append(self.__PAAC__)
                else:
                    rowPAA.append(empty_matrix)
                    rowHAC.append(empty_matrix)
                    rowPAC.append(empty_matrix)
            else:
                blockHAC.append(rowHAC)
                blockPAA.append(rowPAA)
                blockPAC.append(rowPAC)
        ## AB and AD matrix
        for r in range(SU_sep):
            rowHAB = []
            rowHAD = []
            for c in range(SU_ovl):
                if r == c:
                    rowHAB.append(self.__MAB__)
                    rowHAD.append(self.__MAD__)
                elif r == c+1:
                    rowHAB.append(self.__MpAB__)
                    rowHAD.append(self.__MpAD__)
                else:
                    rowHAB.append(empty_matrix)
                    rowHAD.append(empty_matrix)
            else:
                blockHAB.append(rowHAB)
                blockHAD.append(rowHAD)
        ## BC matrix
        for r in range(SU_ovl):
            rowHBC = []
            for c in range(SU_sep):
                if r == c:
                    rowHBC.append(self.__MBC__)
                elif r == c-1:
                    rowHBC.append(self.__MnBC__)
                else:
                    rowHBC.append(empty_matrix)
            else:
                blockHBC.append(rowHBC)
        ## BB and BD matrix
        for r in range(SU_ovl):
            rowHBD = []
            rowHBB = []
            rowPBD = []
            for c in range(SU_ovl):
                if r == c:
                    rowHBD.append(self.__MBD__)
                    rowHBB.append(self.__PAAA__)
                    rowPBD.append(self.__PABD__)
                else:
                    rowHBD.append(empty_matrix)
                    rowHBB.append(empty_matrix)
                    rowPBD.append(empty_matrix)
            else:
                blockHBD.append(rowHBD)
                blockHBB.append(rowHBB)
                blockPBD.append(rowPBD)
        blockHAB = np.block(blockHAB)
        blockHAC = np.block(blockHAC)
        blockHAD = np.block(blockHAD)
        blockHBB = np.block(blockHBB)
        blockHBC = np.block(blockHBC)
        blockHBD = np.block(blockHBD)
        blockPAA = np.block(blockPAA)
        blockPAC = np.block(blockPAC)
        blockPBD = np.block(blockPBD)
        ## combine matrix
        m_ss = np.zeros((self.SU_size*2*SU_sep,self.SU_size*2*SU_sep), dtype=np.complex128)
        m_so = np.zeros((self.SU_size*2*SU_sep,self.SU_size*2*SU_ovl), dtype=np.complex128)
        m_os = np.zeros((self.SU_size*2*SU_ovl,self.SU_size*2*SU_sep), dtype=np.complex128)
        m_oo = np.zeros((self.SU_size*2*SU_ovl,self.SU_size*2*SU_ovl), dtype=np.complex128)
        if self.gap_inv == 1:
            H = [[m_ss, blockHAB],
                 [m_os, blockHBB]]
            P = [[blockPAA, m_so],
                 [m_os, m_oo]]
        else:
            H = [[m_ss,blockHAB,blockHAC,blockHAD],
                 [m_os,blockHBB,blockHBC,blockHBD],
                 [m_ss,m_so,m_ss,blockHAB],
                 [m_os,m_oo,m_os,blockHBB]]
            P = [[blockPAA,m_so,blockPAC,m_so],
                 [m_os,m_oo,m_os,blockPBD],
                 [m_ss,m_so,blockPAA,m_so],
                 [m_os,m_oo,m_os,m_oo]]
        self.H = np.block(H)
        self.H = self.H + np.transpose(np.conj(self.H))
        self.Pb = np.block(P)
        self.Pf = np.transpose(np.conj(self.Pb))
    def __component__(self):
        '''
        1)
        matrix element definition
        |---------------------------|
        |   PRn   |   Mn  |   PAn   |
        |---------------------------|
        |   PR    |   M   |   PA    |
        |---------------------------|
        |   PRp   |   Mp  |   PAp   |
        |---------------------------|
        2)
        M define
        '''
        '''
        ==============================================
                                   b'  a',B'  A'
        SU type: overlap           X----0====O
                                   B   B,D   D
        ==============================================
                                a    A    b    B
        SU type: separate       X    O    X    O
                                A    C    A    C
        ==============================================
        H sequence: a,b,...,a',b',...,A,B,...A',B',...
        '''
        empty_matrix = np.zeros((self.SU_size*2,self.SU_size*2), dtype=np.complex128)
        ## same layer hopping
        self.__MAB__ = copy.deepcopy(empty_matrix)
        self.__MAB__[0,0] = -self.mat.r0
        self.__MAB__[1,1] = -self.mat.r0
        self.__MpAB__ = self.__MAB__
        self.__PAAA__ = copy.deepcopy(empty_matrix)
        self.__PAAA__[0,1] = -self.mat.r0
        ## inter layer hopping
        self.__MAC__ = copy.deepcopy(empty_matrix)
        self.__MAC__[1,0] = -self.mat.r3
        self.__MAD__ = copy.deepcopy(empty_matrix)
        self.__MAD__[1,1] = -self.mat.r3
        self.__MpAD__ = self.__MAD__
        self.__MBC__ = copy.deepcopy(empty_matrix)
        self.__MBC__[0,0] = -self.mat.r3
        self.__MnBC__ = self.__MBC__
        self.__MBD__ = copy.deepcopy(empty_matrix)
        self.__MBD__[1,0] = -self.mat.r1
        self.__PAAC__ = copy.deepcopy(empty_matrix)
        self.__PAAC__[0,1] = -self.mat.r1
        self.__PABD__ = copy.deepcopy(empty_matrix)
        self.__PABD__[0,1] = -self.mat.r3