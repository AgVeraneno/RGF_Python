import copy, os
import numpy as np

class ATNR6():
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
        self.SU_size = 6                        # sub unit cell size (number of hopping and spin)
        self.SU_count = 1                       # atom number for each sub unit cell
        '''
        Auto generate parameters
        '''
        self.mat = setup['material']            # unit cell material
        self.mesh = int(setup['kx_mesh'])       # band structure mesh
        self.ax = self.mat.ax                   # unit length
        self.__initialize__(setup, job)  # build unit cell geometry
        self.__gen_Hamiltonian__()
    def __initialize__(self, setup, job):
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
            self.lattice = 'MLG'
        elif setup['lattice'] == 'BLG':
            self.m_size = 4*sum(self.W)
            self.gap_inv = 0
            self.lattice = 'BLG'
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
            gap = self.mat.A0
            for j, sep in enumerate(range(SU_sep[i])):
                m = self.SU_size*(shift+sep)
                if self.gap_inv:        # MLG
                    for k in range(self.SU_size):
                        gap_profile[m+k,m+k] = gap[k,k]
            else:
                shift += SU_sep[i]
        ## overlap type sub unit
        shift = sum(SU_sep)
        for i in range(len(self.W)):
            gap = self.mat.A0
            for j, sep in enumerate(range(SU_ovl[i])):
                m = self.SU_size*(shift+sep)
                if self.gap_inv:        # MLG
                    for k in range(self.SU_size):
                        gap_profile[m+k,m+k] = gap[k,k]
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
                m = self.SU_size*(shift+sep)
                if self.gap_inv:        # MLG
                    if self.SU_type == 'separate':
                        for k in range(self.SU_size):
                            dv_profile[m+k,m+k] = Vbot+2*(j+0.5)*dV
                    elif self.SU_type == 'overlap':
                        for k in range(self.SU_size):
                            dv_profile[m+k,m+k] = Vbot+2*(j+1)*dV
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
                m = self.SU_size*(shift+sep)
                if self.gap_inv:        # MLG
                    if self.SU_type == 'separate':
                        for k in range(self.SU_size):
                            dv_profile[m+k,m+k] = Vbot+2*(j+1)*dV
                    elif self.SU_type == 'overlap':
                        for k in range(self.SU_size):
                            dv_profile[m+k,m+k] = Vbot+2*(j+0.5)*dV
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
        empty_matrix = np.zeros((self.SU_size,self.SU_size), dtype=np.complex128)
        ## unit cell H
        H = []
        blockHAA = []
        blockHAB = []
        blockHBB = []
        ## forward hopping matrix Pf
        Pf = []
        blockPAB = []
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
        ## AA matrix
        for r in range(SU_sep):
            rowHAA = []
            for c in range(SU_sep):
                if r == c-1:
                    rowHAA.append(self.mat.A4)
                else:
                    rowHAA.append(empty_matrix)
            else:
                blockHAA.append(rowHAA)
        ## AB matrix
        for r in range(SU_sep):
            rowHAB = []
            rowPAB = []
            for c in range(SU_ovl):
                if r == c:
                    rowHAB.append(self.mat.A3)
                    rowPAB.append(self.mat.A5)
                elif r == c+1:
                    rowHAB.append(self.mat.A2)
                    rowPAB.append(self.mat.A6)
                else:
                    rowHAB.append(empty_matrix)
                    rowPAB.append(empty_matrix)
            else:
                blockHAB.append(rowHAB)
                blockPAB.append(rowPAB)
        ## BB matrix
        for r in range(SU_ovl):
            rowHBB = []
            for c in range(SU_ovl):
                if r == c-1:
                    rowHBB.append(self.mat.A4)
                else:
                    rowHBB.append(empty_matrix)
            else:
                blockHBB.append(rowHBB)
        blockHAA = np.block(blockHAA)
        blockHAB = np.block(blockHAB)
        blockHBB = np.block(blockHBB)
        blockPAB = np.block(blockPAB)
        ## combine matrix
        m_ss = np.zeros((self.SU_size*SU_sep,self.SU_size*SU_sep), dtype=np.complex128)
        m_so = np.zeros((self.SU_size*SU_sep,self.SU_size*SU_ovl), dtype=np.complex128)
        m_os = np.zeros((self.SU_size*SU_ovl,self.SU_size*SU_sep), dtype=np.complex128)
        m_oo = np.zeros((self.SU_size*SU_ovl,self.SU_size*SU_ovl), dtype=np.complex128)
        if self.gap_inv == 1:
            H = [[blockHAA, blockHAB],
                 [m_os, blockHBB]]
            P = [[m_ss, blockPAB],
                 [m_os, m_oo]]
        self.H = np.block(H)
        self.H = self.H + np.transpose(np.conj(self.H))
        self.Pf = np.block(P)
        self.Pb = np.transpose(np.conj(self.Pf))
    def __component__(self):
        pass
class ATNR10():
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
        self.SU_size = 10                        # sub unit cell size
        self.SU_count = 1
        '''
        Auto generate parameters
        '''
        self.__initialize__(setup, job)  # build unit cell geometry
        self.mesh = int(setup['kx_mesh'])       # band structure mesh
        self.ax = self.mat.ax                   # unit length
        self.__gen_Hamiltonian__()
    def __initialize__(self, setup, job):
        self.mat = setup['material']                # unit cell material
        self.SU_type = setup['SU_type']             # sub unitcell type
        '''
        matrix definition
        '''
        ## ribbon size
        self.W = job['width']
        self.L = max(job['length'])
        ## lattice type
        if setup['lattice'] == 'MLG':
            self.m_size = self.SU_size*sum(self.W)
            self.gap_inv = 1
        elif setup['lattice'] == 'BLG':
            self.m_size = 2*self.SU_size*sum(self.W)
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
        self.mat.Ez = job['Ez'][0]
        self.mat.Bx = job['Bx'][0]
        self.mat.By = job['By'][0]
        self.mat.Bz = job['Bz'][0]
        if not setup['SOI']:
            self.mat.SOI = 0
        self.mat.setATNR10()
    def __gen_Hamiltonian__(self):
        self.__component__()
        self.__off_diagonal__()
        self.__on_site_energy__()
    def __on_site_energy__(self):
        ## sub unit cell size

        SU = [int(W/2) for W in self.W]
        SU_add = [W%2 for W in self.W]
        W = sum(self.W)
        '''
        Voltage profile
        '''
        dv_profile = np.zeros((self.m_size,self.m_size), dtype=np.complex128)
        if self.SU_type == 'separate':
            isSep = True
        elif self.SU_type == 'overlap':
            isSep = False
        else:
            raise ValueError('Unresolved type:',self.SU_type)
        SU_shift = (int(W/2)+W%2)*self.SU_size
        cnt_sep = 0
        cnt_ovl = 0
        for i, w in enumerate(self.W):
            Vtop = self.Vtop[i]
            Vbot = self.Vbot[i]
            dV = (Vtop - Vbot)/w
            if self.gap_inv:        # MLG
                for j in range(w):
                    if isSep:
                        m = cnt_sep*self.SU_size
                        ## separate part
                        for k in range(self.SU_size):
                            dv_profile[m+k,m+k] = Vbot+(j+0.5)*dV
                        else:
                            isSep = False
                            cnt_sep += 1
                    else:
                        m = SU_shift+cnt_ovl*self.SU_size
                        ## overlap part
                        for k in range(self.SU_size):
                            dv_profile[m+k,m+k] = Vbot+(j+0.5)*dV
                        else:
                            isSep = True
                            cnt_ovl += 1
        '''
        Gap Profile
        '''
        '''
        Combine
        '''
        #self.H += gap_profile
        self.H += dv_profile
        self.V = dv_profile
    def __off_diagonal__(self):
        empty_matrix = np.zeros((self.SU_size,self.SU_size), dtype=np.complex128)
        ## unit cell H
        H = []
        blockHAA = []
        blockHAB = []
        blockHBB = []
        ## forward hopping matrix Pf
        Pf = []
        blockPAB = []
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
        ## AA matrix
        for r in range(SU_sep):
            rowHAA = []
            for c in range(SU_sep):
                if r == c-1:
                    rowHAA.append(self.mat.A4)
                elif r == c:
                    rowHAA.append(self.mat.A0/2)
                else:
                    rowHAA.append(empty_matrix)
            else:
                blockHAA.append(rowHAA)
        ## AB matrix
        for r in range(SU_sep):
            rowHAB = []
            rowPAB = []
            for c in range(SU_ovl):
                if r == c:
                    rowHAB.append(self.mat.A3)
                    rowPAB.append(self.mat.A5)
                elif r == c+1:
                    rowHAB.append(self.mat.A2)
                    rowPAB.append(self.mat.A6)
                else:
                    rowHAB.append(empty_matrix)
                    rowPAB.append(empty_matrix)
            else:
                blockHAB.append(rowHAB)
                blockPAB.append(rowPAB)
        ## BB matrix
        for r in range(SU_ovl):
            rowHBB = []
            for c in range(SU_ovl):
                if r == c-1:
                    rowHBB.append(self.mat.A4)
                elif r == c:
                    rowHBB.append(self.mat.A0/2)
                else:
                    rowHBB.append(empty_matrix)
            else:
                blockHBB.append(rowHBB)
        blockHAA = np.block(blockHAA)
        blockHAB = np.block(blockHAB)
        blockHBB = np.block(blockHBB)
        blockPAB = np.block(blockPAB)
        ## combine matrix
        m_ss = np.zeros((self.SU_size*SU_sep,self.SU_size*SU_sep), dtype=np.complex128)
        m_so = np.zeros((self.SU_size*SU_sep,self.SU_size*SU_ovl), dtype=np.complex128)
        m_os = np.zeros((self.SU_size*SU_ovl,self.SU_size*SU_sep), dtype=np.complex128)
        m_oo = np.zeros((self.SU_size*SU_ovl,self.SU_size*SU_ovl), dtype=np.complex128)
        if self.gap_inv == 1:
            H = [[blockHAA, blockHAB],
                 [m_os, blockHBB]]
            P = [[m_ss, blockPAB],
                 [m_os, m_oo]]
        self.H = np.block(H)
        self.H = self.H + np.transpose(np.conj(self.H))
        self.Pf = np.block(P)
        self.Pb = np.transpose(np.conj(self.Pf))
    def __component__(self):
        pass