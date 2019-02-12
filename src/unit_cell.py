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
        self.mat = setup['material']            # unit cell material
        self.brick_size = int(setup['brick_size'])# sub unitcell size
        self.__structure__(job)                 # build unit cell geometry
        self.mesh = int(setup['kx_mesh'])       # band structure mesh
        self.ax = self.mat.ax                   # unit length
        ## info
        self.info = {'region':job['region'],
                     'lattice':setup['lattice'],
                     'brief':setup['brief']}
        ## on site energy
        self.gap = job['gap']
        self.Vtop = job['Vtop']
        self.Vbot = job['Vbot']
        self.dV = [(job['Vtop'][i]-job['Vbot'][i])/W for i,W in enumerate(job['width'])]
        ## matrix entrance
        self.L1_start = job['shift']
        self.L1_stop = [self.L1_start[i]+W-1 for i,W in enumerate(job['width'])]
        self.L2_start = [self.half_m+S for S in job['shift']]
        self.L2_stop = [self.L2_start[i]+W-1 for i,W in enumerate(job['width'])]
        ## Hamiltonian
        empty_matrix = np.zeros((self.BL_size,self.BL_size), dtype=np.complex128)
        self.H = copy.deepcopy(empty_matrix)
        self.P_plus = copy.deepcopy(empty_matrix)
        self.P_minus = copy.deepcopy(empty_matrix)
        self.__gen_Hamiltonian__(setup['lattice'])
    def __structure__(self, job):
        ## matrix size
        if job['cell_type'] == 'wave':
            self.add_top = False
            self.W = sum(job['width'])
        elif job['cell_type'] == 'envelope':
            self.add_top = True
            self.W = sum(job['width'])+1
            job['width'][-1] += 1
        else:
            raise ValueError('Unresolved cell_type:',job['cell_type'])
        self.half_m = min(job['shift'])+self.W
        self.ML_size = self.half_m*self.brick_size
        self.BL_size = self.half_m*self.brick_size*2
        self.L = job['length']
    def __gen_Hamiltonian__(self, lattice='MLG'):
        ## generate matrix components ##
        self.__4by4Component__()
        ## assemble matrix ##
        self.__off_diagonal__(lattice)
        self.__on_site_energy__(lattice)
        ## resize to half if lattice type is MLG ##
        if lattice == 'MLG':
            self.H = self.H[0:self.ML_size,0:self.ML_size]
            self.P_plus = self.P_plus[0:self.ML_size,0:self.ML_size]
            self.P_minus = self.P_minus[0:self.ML_size,0:self.ML_size]
    def setKx(self, l_idx):
        kx = 2*l_idx*np.pi/(self.ax*self.mesh)
        self.kx_norm = kx*self.ax/np.pi
        return kx
    def __on_site_energy__(self, lattice='MLG'):
        '''
        Width profile
        '''
        gap_profile = np.ones(max(self.L1_stop)+1)*1000
        vtop_profile = np.zeros(max(self.L1_stop)+1)
        vbot_profile = np.zeros(max(self.L1_stop)+1)
        dv_profile = np.zeros(max(self.L1_stop)+1)
        for i in range(len(self.L1_start)):
            l1s = self.L1_start[i]
            l1e = self.L1_stop[i]
            counter = l1s
            while counter <= l1e:
                gap_profile[counter] = self.gap[i]
                vtop_profile[counter] = self.Vtop[i]
                vbot_profile[counter] = self.Vbot[i]
                dv_profile[counter] = self.dV[i]
                counter += 1
        '''
        Generate on site energy
        '''
        for m in range(self.ML_size):
            block = int(m/self.brick_size)        # get current block
            '''
            On site gap energy
            '''
            site_gap = gap_profile[block]
            '''
            Potential
            '''
            if m%4 == 0 or m%4 == 3:
                site_V = (vbot_profile[block]+(block+0.5)*dv_profile[block])
            else:
                site_V = (vbot_profile[block]+block*dv_profile[block])
            ##
            if lattice == 'MLG':
                if block == max(self.L1_stop) and self.add_top:
                    ## top cell ##
                    if m%self.brick_size == 0:
                        self.H[m,m] = 1000
                    elif m%self.brick_size == 1:
                        self.H[m,m] = -site_gap+site_V
                    elif m%self.brick_size == 2:
                        self.H[m,m] = site_gap+site_V
                    else:
                        self.H[m,m] = -1000
                else:
                    if m%2 == 0:
                        self.H[m,m] = site_gap+site_V
                    else:
                        self.H[m,m] = -site_gap+site_V
            elif lattice == 'BLG':
                if block == max(self.L1_stop) and self.add_top:
                    ## top cell ##
                    if m%self.brick_size == 1 or m%self.brick_size == 2:
                        self.H[m,m] = site_gap+site_V
                        self.H[m+self.ML_size,m+self.ML_size] = -site_gap+site_V
                    else:
                        self.H[m,m] = 1000
                        self.H[m+self.ML_size,m+self.ML_size] = -1000
                else:
                    self.H[m,m] = site_gap+site_V
                    self.H[m+self.ML_size,m+self.ML_size] = -site_gap+site_V
    def __off_diagonal__(self, lattice='MLG'):
        empty_matrix = np.zeros((self.brick_size,self.brick_size), dtype=np.complex128)
        l1s = min(self.L1_start)
        l1e = max(self.L1_stop)
        l2s = min(self.L2_start)
        l2e = max(self.L2_stop)
        ## unit cell H
        np_matrix = []
        np_matrixP = []
        ##
        for r in range(2*self.half_m):
            row = []
            rowP = []
            for c in range(2*self.half_m):
                if r >= l1s and r <= l1e:
                    if c >= l1s and c <= l1e:        # AB-AB
                        if r == c:
                            if self.add_top and r == l1e:
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
                    elif c >= l2s and c <= l2e:        # AB-ab
                        if r == c-self.W:
                            if self.add_top and r == l1e:
                                row.append(self.__AB2abtop__)
                                rowP.append(self.__AB2abtop_P__)
                            else:
                                row.append(self.__AB2ab__)
                                rowP.append(self.__AB2ab_P__)
                        elif r == c-self.W-1:
                            row.append(self.__AB2abnext__)
                            rowP.append(empty_matrix)
                        elif r == c-self.W+1 and r != l2s:
                            row.append(self.__AB2abpre__)
                            rowP.append(empty_matrix)
                        else:
                            row.append(empty_matrix)
                            rowP.append(empty_matrix)
                    else:
                        row.append(empty_matrix)
                        rowP.append(empty_matrix)
                elif r >= l2s and r <= l2e:
                    if c >= l2s and c <= l2e:        # ab-ab
                        if r == c:
                            if self.add_top and r == l2e:
                                row.append(self.__ab2abtop__)
                                rowP.append(empty_matrix)
                            else:
                                row.append(self.__ab2ab__)
                                rowP.append(self.__ab2ab_P__)
                        elif r == c-1:
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
        empty_matrix = np.zeros((self.brick_size,self.brick_size), dtype=np.complex128)
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