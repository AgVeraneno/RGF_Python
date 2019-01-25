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
        self.mat = setup['material']
        self.subunit_size = 4
        ## matrix size
        if job['cell_type'] == 'wave':
            self.add_top = False
            subunit_count = int(job['width'])
        elif job['cell_type'] == 'envelope':
            self.add_top = True
            subunit_count = int(job['width'])+1
        else:
            raise ValueError('Unresolved cell_type:',job['cell_type'])
        MLG_m_size = subunit_count*4
        BLG_m_size = subunit_count*8
        ## on site energy
        self.gap = float(job['gap'])
        self.Vtop = float(job['Vtop'])
        self.Vbot = float(job['Vbot'])
        self.dV = (self.Vtop-self.Vbot)/(subunit_count-1)
        ## matrix entrance
        self.L1_start = int(job['shift'])
        self.L1_stop = self.L1_start+subunit_count-1
        self.L2_start = self.L1_stop+int(job['shift'])+1
        self.L2_stop = self.L2_start+subunit_count-1
        ## Hamiltonian
        empty_matrix = np.zeros((BLG_m_size,BLG_m_size), dtype=np.complex128)
        self.H = copy.deepcopy(empty_matrix)
        self.P_plus = copy.deepcopy(empty_matrix)
        self.P_minus = copy.deepcopy(empty_matrix)
        self.__gen_Hamiltonian__(setup['lattice'])
    def __gen_Hamiltonian__(self, lattice='MLG'):
        self.__4by4Component__()
        self.__off_diagonal__(lattice)
        self.__on_site_energy__(lattice)
    def setKx(self, l_idx):
        self.kx = 2*l_idx*np.pi/(self.mat.ax*self.inputs['kx_mesh'])
        self.kx_norm = self.kx*self.mat.ax/np.pi
    def __on_site_energy__(self, lattice='MLG'):
        full_matrix_size = self.L2_stop+1
        half_matrix_size = int(full_matrix_size/2)
        actual_size = half_matrix_size*self.subunit_size
        ## place gap open and transverse field ##
        if lattice == 'MLG':
            for m in range(actual_size):
                block = int(m/self.subunit_size)        # get current block
                ## define gap and V
                # assign A as +1 and B as -1
                site_gap = -m%2*self.gap + (1-m%2)*self.gap
                # on site V on sub unit cell
                site_V = (self.Vbot+(block-self.L1_start)*self.dV)
                if block >= self.L1_start and block <= self.L1_stop:
                    if block == self.L1_stop and self.add_top:
                        if m%self.subunit_size == 0:
                            self.H[m,m] = 1000
                        elif m%self.subunit_size == 3:
                            self.H[m,m] = -1000
                        else:
                            self.H[m,m] = site_gap+site_V
                    else:
                        self.H[m,m] = site_gap+site_V
                else:
                    self.H[m,m] = 1000
        elif lattice == 'BLG':
            for m in range(actual_size):
                block = int(m/self.subunit_size)        # get current block
                ## define gap and V
                # assign A as +1 and B as -1
                site_gap = self.gap
                # on site V on sub unit cell
                site_V = (self.Vbot+(block-self.L1_start)*self.dV)
                if block >= self.L1_start and block <= self.L1_stop:
                    if block == self.L1_stop and self.add_top:
                        if m%self.subunit_size == 0 or m%self.subunit_size == 3:
                            self.H[m,m] = 1000
                            self.H[m+actual_size,
                                   m+actual_size] = -1000
                        else:
                            self.H[m,m] = site_gap+site_V
                            self.H[m+actual_size,
                                   m+actual_size] = -site_gap+site_V
                    else:
                        self.H[m,m] = site_gap+site_V
                        self.H[m+actual_size,
                               m+actual_size] = -site_gap+site_V
                else:
                    self.H[m,m] = 1000
                    self.H[m+actual_size,
                           m+actual_size] = -1000
    def __off_diagonal__(self, lattice='MLG'):
        empty_matrix = np.zeros((self.subunit_size,self.subunit_size), dtype=np.complex128)
        full_matrix_size = self.L2_stop+1
        half_matrix_size = int(full_matrix_size/2)
        ## unit cell H
        np_matrix = []
        np_matrixP = []
        for r in range(full_matrix_size):
            row = []
            rowP = []
            for c in range(full_matrix_size):
                if r >= self.L1_start and r <= self.L1_stop:
                    if c >= self.L1_start and c <= self.L1_stop:        # AB-AB
                        if r == c:
                            if self.add_top and r == self.L1_stop:
                                row.append(self.__AB2ABtop__)
                                rowP.append(empty_matrix)
                            else:
                                row.append(self.__AB2AB__)
                                rowP.append(self.__AB2AB_P__)
                        elif r == c-1:
                            row.append(self.__AB2ABnext__)
                            rowP.append(empty_matrix)
                        else:
                            row.append(empty_matrix)
                            rowP.append(empty_matrix)
                    elif c >= self.L2_start and c <= self.L2_stop:        # AB-ab
                        if r == c-half_matrix_size:
                            if self.add_top and r == self.L1_stop:
                                row.append(self.__AB2abtop__)
                                rowP.append(self.__AB2abtop_P__)
                            else:
                                row.append(self.__AB2ab__)
                                rowP.append(self.__AB2ab_P__)
                        elif r == c-half_matrix_size-1:
                            row.append(self.__AB2abnext__)
                            rowP.append(empty_matrix)
                        elif r == c-half_matrix_size+1 and r != self.L2_start:
                            row.append(self.__AB2abpre__)
                            rowP.append(empty_matrix)
                        else:
                            row.append(empty_matrix)
                            rowP.append(empty_matrix)
                    else:
                        row.append(empty_matrix)
                        rowP.append(empty_matrix)
                elif r >= self.L2_start and r <= self.L2_stop:
                    if c >= self.L2_start and c <= self.L2_stop:        # ab-ab
                        if r == c:
                            if self.add_top and r == self.L2_stop:
                                row.append(self.__ab2abtop__)
                                rowP.append(empty_matrix)
                            else:
                                row.append(self.__ab2ab__)
                                rowP.append(self.__ab2ab_P__)
                        elif r == c+1:
                            row.append(self.__ab2abnext__)
                            rowP.append(empty_matrix)
                        else:
                            row.append(empty_matrix)
                            rowP.append(empty_matrix)
                    else:
                        row.append(empty_matrix)
                        rowP.append(empty_matrix)
                else:
                    row.append(empty_matrix)
                    rowP.append(empty_matrix)
            np_matrix.append(row)
            np_matrixP.append(rowP)
        self.H = np.block(np_matrix)
        self.H = self.H + np.transpose(np.conj(self.H))
        self.P_minus = np.block(np_matrixP)
        self.P_plus = np.transpose(np.conj(self.P_minus))
    def __4by4Component__(self):
        '''
        =================================
                           b   a',B  A'
        top edge           X-----0   O  
        =================================
                        a    A    b'   B'
                        X    O    X   O
        subunit cell     \       /   
                          X-----0   O
                          b   a',B  A'
        =================================
        H sequence: A1, B1, A1', B1', A2, B2, ..., Am', Bm', a1, b1, a1', b1', a2, b2, ..., am, bm
        '''
        empty_matrix = np.zeros((self.subunit_size,self.subunit_size), dtype=np.complex128)
        # within sub unit cell hopping
        self.__AB2AB__ = copy.deepcopy(empty_matrix)
        self.__ab2ab__ = copy.deepcopy(empty_matrix)
        self.__AB2ABtop__ = copy.deepcopy(empty_matrix)
        self.__ab2abtop__ = copy.deepcopy(empty_matrix)
        # next sub unit cell hopping
        self.__AB2ABnext__ = copy.deepcopy(empty_matrix)
        self.__ab2abnext__ = copy.deepcopy(empty_matrix)
        # within sub unit cell inter layer hopping
        self.__AB2ab__ = copy.deepcopy(empty_matrix)
        self.__AB2abtop__ = copy.deepcopy(empty_matrix)
        # next sub unit cell inter layer hopping
        self.__AB2abnext__ = copy.deepcopy(empty_matrix)
        self.__AB2abpre__ = copy.deepcopy(empty_matrix)
        self.__AB2abpretop__ = copy.deepcopy(empty_matrix)
        # next unit cell, same sub unit cell, same layer hopping
        self.__AB2AB_P__ = copy.deepcopy(empty_matrix)
        self.__ab2ab_P__ = copy.deepcopy(empty_matrix)
        # next unit cell, same sub unit cell, different layer hopping
        self.__AB2ab_P__ = copy.deepcopy(empty_matrix)
        self.__AB2abtop_P__ = copy.deepcopy(empty_matrix)
        '''
        AB/ab intra and inter layer hopping
              Hsu                 P-                 AB->ab               P- AB->ab
        a   b   a'  b'
        ==============       ==============       ==============       ==============
        A   B   A'  B'
        ==============       ==============       ==============       ==============
        v   r0  0   0        0   0   0  r0        0   r3  0   r3       0   0   0   0
        -   v   r0  0        0   0   0   0        0   0   r1  0        0   0   0   0
        -   -   v   r0       0   0   0   0        0   0   0   r3       0   r3  0   0
        -   -   -   v        0   0   0   0        0   0   0   0        r1  0   0   0
        ==============      ==============        ==============       ==============
        '''
        # H
        self.__AB2AB__[0,1] = -self.mat.r0
        self.__AB2AB__[1,2] = -self.mat.r0
        self.__AB2AB__[2,3] = -self.mat.r0
        self.__ab2ab__[0,1] = -self.mat.r0
        self.__ab2ab__[1,2] = -self.mat.r0
        self.__ab2ab__[2,3] = -self.mat.r0
        self.__AB2ab__[0,1] = -self.mat.r3
        self.__AB2ab__[0,3] = -self.mat.r3
        self.__AB2ab__[1,2] = -self.mat.r1
        self.__AB2ab__[2,3] = -self.mat.r3
        self.__AB2ABtop__[1,2] = -self.mat.r0
        self.__ab2abtop__[1,2] = -self.mat.r0
        self.__AB2abtop__[1,2] = -self.mat.r1
        # P-
        self.__AB2AB_P__[0,3] = -self.mat.r0
        self.__ab2ab_P__[0,3] = -self.mat.r0
        self.__AB2ab_P__[2,1] = -self.mat.r3
        self.__AB2ab_P__[3,0] = -self.mat.r1
        self.__AB2abtop_P__[2,1] = -self.mat.r3
        '''
        AB/ab intra and inter subcell hopping
           Hsu next           AB->ab next
        a   b   a'  b'
        ==============       ==============
        A   B   A'  B'
        ==============       ==============
        0   r0  0   0        0   r3  0   0
        0   0   0   0        0   0   0   0
        0   0   0   0        0   0   0   0
        0   0   r0  0        0   0   0   0
        ==============
        '''
        # H
        self.__AB2ABnext__[0,1] = -self.mat.r0
        self.__AB2ABnext__[3,2] = -self.mat.r0
        self.__ab2abnext__[0,1] = -self.mat.r0
        self.__ab2abnext__[3,2] = -self.mat.r0
        self.__AB2abnext__[0,1] = -self.mat.r3
        '''
        AB/ab intra and inter subcell hopping
          AB->ab pre
        a   b   a'  b'
        ==============
        A   B   A'  B'
        ==============
        0   0   0   0
        0   0   0   0
        0   0   0   r3
        0   0   0   0
        ==============
        '''
        # H
        self.__AB2abpre__[2,3] = -self.mat.r3
        self.__AB2abpretop__[2,3] = -self.mat.r3