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
        self.SU_count = 2                       # atom number for each sub unit cell
        '''
        Auto generate parameters
        '''
        self.mat = setup['material']            # unit cell material
        self.mesh = int(setup['kx_mesh'])       # band structure mesh
        self.ax = self.mat.ax                   # unit length
        self.__initialize__(setup, job)
        self.__gen_Hamiltonian__()
    def __initialize__(self, setup, job):
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
        elif setup['lattice'] == 'BLG':
            self.m_size = 4*sum(self.W)
            self.gap_inv = 0
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
        ## identify index
        idx_list = []
        sep_count = int(sum(self.W)/2) + sum(self.W)%2
        ovl_count = sum(self.W) - sep_count
        for s in range(sep_count):
            for i in range(self.SU_size):
                idx_list.append(2*s+i)
                idx_list.append(2*s+i+sum(self.W))
        for o in range(ovl_count):
            for i in range(self.SU_size):
                idx_list.append(2*o+1+i)
                idx_list.append(2*o+1+sum(self.W)+i)
        '''
        Gap Profile
        '''
        gap_profile = np.eye(self.m_size, dtype=np.complex128)*1000
        gap_list = []
        counter = 0
        for w_idx, w in enumerate(self.W):
            if self.gap_inv:
                for i in range(w):
                    gap_list.append(self.gap[w_idx]*(-1)**counter)
                    counter += 1
            else:
                for i in range(w):
                    gap_list.append(self.gap[w_idx])
        else:
            gap_inv = [g*-1 for g in gap_list]
            gap_list.extend(gap_inv)
            for i_idx, idx in enumerate(idx_list):
                gap = gap_list[i_idx]
                gap_profile[idx, idx] = gap
        ## sub unit cell size
        SU = [int(W/2) for W in self.W]
        sep_add = [W%2 for W in self.W]
        ovl_add = [W%2 for W in self.W]
        for idx in range(1,len(sep_add)):
            sep_add[idx] *= sep_add[idx-1]
        W = 2*sum(self.W)
        SU_sep = [SU[i] + sep_add[i] for i in range(len(self.W))]
        SU_ovl = [SU[i] + ovl_add[i] - sep_add[i] for i in range(len(self.W))]
        SU_shift = sum(SU_sep)
        '''
        Voltage profile
        '''
        dv_profile = np.zeros((self.m_size,self.m_size), dtype=np.complex128)
        ## separate type sub unit
        shift = 0
        ovl_first = 0
        for i in range(len(self.W)):
            Vtop = self.Vtop[i]
            Vbot = self.Vbot[i]
            dV = (Vtop - Vbot)/self.W[i]
            for j, sep in enumerate(range(SU_sep[i])):
                m = 2*(shift+sep)
                if self.gap_inv:        # MLG
                    dv_profile[m,m] = Vbot+2*(j+0.5+0.5*ovl_first)*dV
                    dv_profile[m+1,m+1] = Vbot+2*(j+0.5+0.5*ovl_first)*dV
                else:       # BLG
                    dv_profile[m,m] = Vbot+2*(j+0.5)*dV
                    dv_profile[m+1,m+1] = Vbot+2*(j+0.5)*dV
                    dv_profile[m+W,m+W] = Vbot+2*(j+0.5)*dV
                    dv_profile[m+W+1,m+W+1] = Vbot+2*(j+0.5)*dV
            else:
                shift += SU_sep[i]
                ovl_first += self.W[i]%2
                ovl_first = ovl_first%2
        ## overlap type sub unit
        shift = sum(SU_sep)
        for i in range(len(self.W)):
            Vtop = self.Vtop[i]
            Vbot = self.Vbot[i]
            dV = (Vtop - Vbot)/self.W[i]
            for j, sep in enumerate(range(SU_ovl[i])):
                m = 2*(shift+sep)
                if self.gap_inv:        # MLG
                    dv_profile[m,m] = Vbot+2*(j+1-0.5*ovl_first)*dV
                    dv_profile[m+1,m+1] = Vbot+2*(j+1-0.5*ovl_first)*dV
                else:       # BLG
                    dv_profile[m,m] = Vbot+2*(j+1)*dV
                    dv_profile[m+1,m+1] = Vbot+2*(j+1)*dV
                    dv_profile[m+W,m+W] = Vbot+2*(j+1)*dV
                    dv_profile[m+W+1,m+W+1] = Vbot+2*(j+1)*dV
            else:
                shift += SU_ovl[i]
                ovl_first += self.W[i]%2
                ovl_first = ovl_first%2
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
        SU_ovl = int(sum(self.W)/2) # overlap sub unitcell count
        SU_sep = sum(self.W) - SU_ovl
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