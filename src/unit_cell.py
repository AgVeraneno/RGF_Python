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
        self.dV = [(job['Vtop'][i]-job['Vbot'][i])/(2*W) for i,W in enumerate(job['width'])]
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
        self.__off_diagonal__()
        self.__on_site_energy__(lattice)
        ## resize to half if lattice type is MLG ##
        if lattice == 'MLG':
            self.H = self.H[0:self.ML_size,0:self.ML_size]
            self.P_plus = self.P_plus[0:self.ML_size,0:self.ML_size]
            self.P_minus = self.P_minus[0:self.ML_size,0:self.ML_size]
            self.H_lead = self.H_lead[0:self.ML_size,0:self.ML_size]
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
        dv_raw = np.zeros(max(self.L1_stop)+1)
        for i in range(len(self.L1_start)):
            l1s = self.L1_start[i]
            l1e = self.L1_stop[i]
            counter = l1s
            while counter <= l1e:
                gap_profile[counter] = self.gap[i]
                vtop_profile[counter] = self.Vtop[i]
                vbot_profile[counter] = self.Vbot[i]
                if self.dV[i] != 0:
                    if counter != 0:
                        dv_profile[counter] = dv_profile[counter-1] + 2*self.dV[i]
                    else:
                        dv_profile[counter] = self.dV[i]
                else:
                    dv_profile[counter] = self.dV[i]
                dv_raw[counter] = self.dV[i]
                counter += 1
        '''
        Generate on site energy
        '''
        self.H_lead = copy.deepcopy(self.H)
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
                site_V = (vbot_profile[block]+dv_profile[block])
            else:
                site_V = (vbot_profile[block]+dv_profile[block]-dv_raw[block])
            ##
            if lattice == 'MLG':
                if block == max(self.L1_stop) and self.add_top:
                    ## top cell ##
                    if m%self.brick_size == 0:
                        self.H[m,m] = 1000
                        self.H_lead[m,m] = 1000
                    elif m%self.brick_size == 1:
                        self.H[m,m] = -site_gap+site_V
                        self.H_lead[m,m] = -site_gap
                    elif m%self.brick_size == 2:
                        self.H[m,m] = site_gap+site_V
                        self.H_lead[m,m] = site_gap
                    else:
                        self.H[m,m] = -1000
                        self.H_lead[m,m] = -1000
                else:
                    if m%2 == 0:
                        self.H[m,m] = site_gap+site_V
                        self.H_lead[m,m] = site_gap
                    else:
                        self.H[m,m] = -site_gap+site_V
                        self.H_lead[m,m] = -site_gap
            elif lattice == 'BLG':
                if block == max(self.L1_stop) and self.add_top:
                    ## top cell ##
                    if m%self.brick_size == 1 or m%self.brick_size == 2:
                        self.H[m,m] = site_gap+site_V
                        self.H[m+self.ML_size,m+self.ML_size] = -site_gap+site_V
                        self.H_lead[m,m] = site_gap
                        self.H_lead[m+self.ML_size,m+self.ML_size] = -site_gap
                    else:
                        self.H[m,m] = 1000
                        self.H[m+self.ML_size,m+self.ML_size] = -1000
                        self.H_lead[m,m] = 1000
                        self.H_lead[m+self.ML_size,m+self.ML_size] = -1000
                else:
                    self.H[m,m] = site_gap+site_V
                    self.H[m+self.ML_size,m+self.ML_size] = -site_gap+site_V
                    self.H_lead[m,m] = site_gap
                    self.H_lead[m+self.ML_size,m+self.ML_size] = -site_gap
    def __off_diagonal__(self):
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
        self.P_plus = np.block(np_matrixP)
        self.P_minus = np.transpose(np.conj(self.P_plus))
    def __4by4Component__(self):
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
class AGNR_new():
    '''
    Unit cell object contain essential information of a unit cell
    H: Hamiltonian of the unit cell containing on site energy & interhopping.
    P_plus: forward inter unit cell hopping matrix
    P_minus: backward inter unit cell hopping matrix
    '''
    def __init__(self, setup, job):
        self.__initialize__(setup, job)         # build unit cell geometry
        self.mesh = int(setup['kx_mesh'])       # band structure mesh
        self.ax = self.mat.ax                   # unit length
        self.__gen_Hamiltonian__(setup['lattice'])
    def __initialize__(self, setup, job):
        self.mat = setup['material']                # unit cell material
        self.SU_type = setup['SU_type']             # sub unitcell type
        self.SU_size = 2*int(setup['SU_hopping_size'])
        '''
        matrix definition
        '''
        ## ribbon size
        for t_idx, t in enumerate(job['type']):
            if t == 'envelope':
                job['width'][t_idx] = 2*job['width'][t_idx]+1
            elif t == 'wave':
                job['width'][t_idx] = 2*job['width'][t_idx]
            else:
                raise ValueError('Unresolved cell type:', t)
        else:
            self.W = job['width']
            self.L = job['length']
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
    def __gen_Hamiltonian__(self, lattice='MLG'):
        self.__component__()
        self.__off_diagonal__()
        self.__on_site_energy__()
    def __on_site_energy__(self):
        '''
        Gap Profile
        '''
        gap_profile = np.ones(sum(self.W)*self.SU_size)*1000
        for i, idx in enumerate(self.L1_start):
            gap = self.gap[i]
            if self.SU_type == 'separate':
                for j in range(self.W[i]):
                    if self.gap_inv:
                        gap_profile[idx+j] = gap-2*(idx+j)%2
                        gap_profile[idx+j+self.W-1] = -gap+2*(idx+j)%2
                    else:
                        gap_profile[idx+2*j+1] = gap
                        gap_profile[idx+j+self.W-1] = gap
                        gap_profile[idx+j+2*self.W-1] = -gap
                        gap_profile[idx+j+3*self.W-1] = -gap
        '''
        Voltage profile
        '''
        dv_profile = np.zeros(sum(self.W)*self.SU_size)
        for i, idx in enumerate(self.L1_start):
            Vtop = self.Vtop[i]
            Vbot = self.Vbot[i]
            W = self.W[i]
            dV = (Vtop - Vbot)/(W+1)
            for j in range(W):
                if self.gap_inv:
                    dv_profile[idx+j] = Vbot + dV*(j+1)
                    dv_profile[idx+j+self.W-1] = Vbot + dV*(j+1)
                else:
                    dv_profile[idx+j] = Vbot + dV*(j+1)
                    dv_profile[idx+j+self.W-1] = Vbot + dV*(j+1)
                    dv_profile[idx+j+2*self.W-1] = Vbot + dV*(j+1)
                    dv_profile[idx+j+3*self.W-1] = Vbot + dV*(j+1)
        '''
        Generate on site energy
        '''
        self.H_lead = copy.deepcopy(self.H)
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
                site_V = (vbot_profile[block]+dv_profile[block])
            else:
                site_V = (vbot_profile[block]+dv_profile[block]-dv_raw[block])
            ##
            if lattice == 'MLG':
                if block == max(self.L1_stop) and self.add_top:
                    ## top cell ##
                    if m%self.brick_size == 0:
                        self.H[m,m] = 1000
                        self.H_lead[m,m] = 1000
                    elif m%self.brick_size == 1:
                        self.H[m,m] = -site_gap+site_V
                        self.H_lead[m,m] = -site_gap
                    elif m%self.brick_size == 2:
                        self.H[m,m] = site_gap+site_V
                        self.H_lead[m,m] = site_gap
                    else:
                        self.H[m,m] = -1000
                        self.H_lead[m,m] = -1000
                else:
                    if m%2 == 0:
                        self.H[m,m] = site_gap+site_V
                        self.H_lead[m,m] = site_gap
                    else:
                        self.H[m,m] = -site_gap+site_V
                        self.H_lead[m,m] = -site_gap
            elif lattice == 'BLG':
                if block == max(self.L1_stop) and self.add_top:
                    ## top cell ##
                    if m%self.brick_size == 1 or m%self.brick_size == 2:
                        self.H[m,m] = site_gap+site_V
                        self.H[m+self.ML_size,m+self.ML_size] = -site_gap+site_V
                        self.H_lead[m,m] = site_gap
                        self.H_lead[m+self.ML_size,m+self.ML_size] = -site_gap
                    else:
                        self.H[m,m] = 1000
                        self.H[m+self.ML_size,m+self.ML_size] = -1000
                        self.H_lead[m,m] = 1000
                        self.H_lead[m+self.ML_size,m+self.ML_size] = -1000
                else:
                    self.H[m,m] = site_gap+site_V
                    self.H[m+self.ML_size,m+self.ML_size] = -site_gap+site_V
                    self.H_lead[m,m] = site_gap
                    self.H_lead[m+self.ML_size,m+self.ML_size] = -site_gap
    def __off_diagonal__(self):
        empty_matrix = np.zeros((self.SU_size,self.SU_size), dtype=np.complex128)
        ## unit cell H
        H = []
        Pf = []
        l1s = self.L1_start[0]
        l1e = self.L1_stop[-1]
        blockHAA = []
        blockHAB = []
        blockHAC = []
        blockHBC = []
        blockHBD = []
        blockPB = []
        blockPD = []
        for r in range(sum(self.W)):
            rowHAA = []
            rowHAB = []
            rowHAC = []
            rowHBC = []
            rowHBD = []
            rowPB = []
            rowPD = []
            for c in range(sum(self.W)):
                if r >= l1s and r <= l1e:
                    if c >= l1s and c <= l1e:
                        if r == c:
                            rowHAA.append(self.__MAA__)
                            rowHAB.append(self.__MAB__)
                            rowHAC.append(self.__MAC__)
                            rowHBC.append(self.__MBC__)
                            rowHBD.append(self.__MBD__)
                            rowPB.append(self.__PAAB__)
                            rowPD.append(self.__PAAD__)
                        elif r == c-1:
                            rowHAA.append(self.__MnAA__)
                            rowHAB.append(empty_matrix)
                            rowHAC.append(self.__MnAC__)
                            rowHBC.append(empty_matrix)
                            rowHBD.append(empty_matrix)
                            rowPB.append(empty_matrix)
                            rowPD.append(empty_matrix)
                        elif r == c+1:
                            rowHAA.append(empty_matrix)
                            rowHAB.append(empty_matrix)
                            rowHAC.append(empty_matrix)
                            rowHBC.append(empty_matrix)
                            rowHBD.append(self.__MpBD__)
                            rowPB.append(empty_matrix)
                            rowPD.append(empty_matrix)
                        else:
                            rowHAA.append(empty_matrix)
                            rowHAB.append(empty_matrix)
                            rowHAC.append(empty_matrix)
                            rowHBC.append(empty_matrix)
                            rowHBD.append(empty_matrix)
                            rowPB.append(empty_matrix)
                            rowPD.append(empty_matrix)
                    else:
                        rowHAA.append(empty_matrix)
                        rowHAB.append(empty_matrix)
                        rowHAC.append(empty_matrix)
                        rowHBC.append(empty_matrix)
                        rowHBD.append(empty_matrix)
                        rowPB.append(empty_matrix)
                        rowPD.append(empty_matrix)
                else:
                    rowHAA.append(empty_matrix)
                    rowHAB.append(empty_matrix)
                    rowHAC.append(empty_matrix)
                    rowHBC.append(empty_matrix)
                    rowHBD.append(empty_matrix)
                    rowPB.append(empty_matrix)
                    rowPD.append(empty_matrix)
            blockHAA.append(rowHAA)
            blockHAB.append(rowHAB)
            blockHAC.append(rowHAC)
            blockHBC.append(rowHBC)
            blockHBD.append(rowHBD)
            blockPB.append(rowPB)
            blockPD.append(rowPD)
        else:
            blockHAA = np.block(blockHAA)
            blockHAB = np.block(blockHAB)
            blockHAC = np.block(blockHAC)
            blockHBC = np.block(blockHBC)
            blockHBD = np.block(blockHBD)
            blockPB = np.block(blockPB)
            blockPD = np.block(blockPD)
        ## combine matrix
        empty_matrix = np.zeros((self.SU_size*sum(self.W),self.SU_size*sum(self.W)), dtype=np.complex128)
        if self.gap_inv == 1:
            H = [[blockHAA      , blockHAB],
                 [empty_matrix, blockHAA]]
            P = [[empty_matrix, blockPB],
                 [empty_matrix, empty_matrix]]
        else:
            H = [[blockHAA,blockHAB,blockHAC,empty_matrix],
                 [empty_matrix,blockHAA,blockHBC,blockHBD],
                 [empty_matrix,empty_matrix,blockHAA,blockHAB],
                 [empty_matrix,empty_matrix,empty_matrix,blockHAA]]
            P = [[empty_matrix,blockPB,empty_matrix,blockPD],
                 [empty_matrix,empty_matrix,empty_matrix,empty_matrix],
                 [empty_matrix,empty_matrix,empty_matrix,blockPB],
                 [empty_matrix,empty_matrix,empty_matrix,empty_matrix]]
        self.H = np.block(H)
        self.H = self.H + np.transpose(np.conj(self.H))
        self.Pf = np.block(P)
        self.Pb = np.transpose(np.conj(self.Pf))
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
                                   A   B,C   D
        ==============================================
                                a    A    b    B
        SU type: separate       X    O    X    O
                                A    C    B    D
        ==============================================
        H sequence: a,b',...,b,a',...,A,B',...B,A',...
        '''
        empty_matrix = np.zeros((self.SU_size,self.SU_size), dtype=np.complex128)
        ## same layer hopping
        self.__MAA__ = copy.deepcopy(empty_matrix)
        self.__MAA__[0,1] = -self.mat.r0
        self.__MnAA__ = copy.deepcopy(empty_matrix)
        self.__MnAA__[1,0] = -self.mat.r0
        self.__MAB__ = copy.deepcopy(empty_matrix)
        self.__MAB__[1,1] = -self.mat.r0
        self.__PAAB__ = copy.deepcopy(empty_matrix)
        self.__PAAB__[0,0] = -self.mat.r0
        ## inter layer hopping
        self.__MAC__ = copy.deepcopy(empty_matrix)
        self.__MAC__[1,0] = -self.mat.r3
        self.__MnAC__ = self.__MAC__
        self.__MBC__ = copy.deepcopy(empty_matrix)
        self.__MBC__[0,0] = -self.mat.r3
        self.__MBC__[1,1] = -self.mat.r1
        self.__PAAD__ = copy.deepcopy(empty_matrix)
        self.__PAAD__[0,0] = -self.mat.r1
        self.__PAAD__[1,1] = -self.mat.r3
        self.__MBD__ = copy.deepcopy(empty_matrix)
        self.__MBD__[0,1] = -self.mat.r3
        self.__MpBD__ = self.__MBD__
class AMNR():
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
        self.__off_diagonal__()
        self.__on_site_energy__(lattice)
        ## resize to half if lattice type is MLG ##
        if lattice == 'MLG':
            self.H = self.H[0:self.ML_size,0:self.ML_size]
            self.P_plus = self.P_plus[0:self.ML_size,0:self.ML_size]
            self.P_minus = self.P_minus[0:self.ML_size,0:self.ML_size]
    def __on_site_energy__(self, lattice='MLG'):
        '''
        Width profile
        '''
        gap_profile = np.ones(max(self.L1_stop)+1)*1000
        vtop_profile = np.zeros(max(self.L1_stop)+1)
        vbot_profile = np.zeros(max(self.L1_stop)+1)
        dv_profile = np.zeros(max(self.L1_stop)+1)
        dv_raw = np.zeros(max(self.L1_stop)+1)
        for i in range(len(self.L1_start)):
            l1s = self.L1_start[i]
            l1e = self.L1_stop[i]
            counter = l1s
            while counter <= l1e:
                gap_profile[counter] = self.gap[i]
                vtop_profile[counter] = self.Vtop[i]
                vbot_profile[counter] = self.Vbot[i]
                if self.dV[i] != 0:
                    if counter != 0:
                        dv_profile[counter] = dv_profile[counter-1] + 2*self.dV[i]
                    else:
                        dv_profile[counter] = self.dV[i]
                else:
                    dv_profile[counter] = self.dV[i]
                dv_raw[counter] = self.dV[i]
                counter += 1
        '''
        Generate on site energy
        '''
        for m in range(self.ML_size):
            half_brick = int(self.brick_size/2)
            block = int(m/self.brick_size)        # get current block
            '''
            On site gap energy
            '''
            site_gap = self.mat.A0[m%half_brick,m%half_brick]
            '''
            Potential
            '''
            if m%self.brick_size >= half_brick:
                site_V = (vbot_profile[block]+dv_profile[block])
            else:
                site_V = (vbot_profile[block]+dv_profile[block]-dv_raw[block])
            ##
            if lattice == 'MLG':
                if block == max(self.L1_stop) and self.add_top:
                    ## top cell ##
                    if m%self.brick_size >= half_brick:
                        self.H[m,m] = 1000
                    else:
                        self.H[m,m] = site_gap+site_V
                else:
                    self.H[m,m] = site_gap+site_V
            elif lattice == 'BLG':
                raise ValueError("Not support BLG for MoS2.")
    def __off_diagonal__(self):
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
                    if c >= l1s and c <= l1e:       # layer 1
                        if r == c:      # diagonal terms
                            if self.add_top and r == l1e:       # top layer
                                row.append(empty_matrix)
                                rowP.append(empty_matrix)
                            else:
                                row.append(self.su_M)
                                rowP.append(self.su_PA)
                        elif r == c-1:  # next sub-cell
                            if self.add_top and c == l1e:       # top layer
                                row.append(self.top_Mn)
                                rowP.append(self.top_PAn)
                            else:
                                row.append(self.su_Mn)
                                rowP.append(self.su_PAn)
                        elif r == c+1 and r != l1s:  # previous sub-cell
                            row.append(empty_matrix)
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
        self.P_plus = np.block(np_matrixP)
        self.P_minus = np.transpose(np.conj(self.P_plus))
    def __4by4Component__(self):
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
        =================================
                                 A
        top edge                 O  
        =================================
                                       B
                                       O
        subunit cell                /        ---> intra subcell
                                 O
                                 A
        =================================
        H sequence: A1, B1, A2, B2, ...
        '''
        '''
        AB/ab intra subcell hopping
              M                    PR                   PA
        ==============       ==============       ==============
        A   B
        ==============       ==============       ==============
        A0  A3               0   A5               0   0
        -   A0               0   0                A2  0
        ==============      ==============        ==============
        
              Mn                  PRp                   PAn
        ==============       ==============       ==============
        A   B
        ==============       ==============       ==============
        A4  0                0   A6               0   0
        A5  A4               0   0                A3  0
        ==============       ==============       ==============
        '''
        mat_z = np.zeros((int(self.brick_size/2),int(self.brick_size/2)), dtype=np.complex128)
        # H
        self.su_M = np.block([[mat_z,self.mat.A3],[mat_z,mat_z]])
        self.su_Mn = np.block([[self.mat.A4,mat_z],[self.mat.A5,self.mat.A4]])
        self.top_Mn = np.block([[self.mat.A4,mat_z],[self.mat.A5,mat_z]])
        # P+
        self.su_PA = np.block([[mat_z,mat_z],[self.mat.A2,mat_z]])
        self.su_PAn = np.block([[mat_z,mat_z],[self.mat.A3,mat_z]])
        self.top_PAn = np.block([[mat_z,mat_z],[self.mat.A3,mat_z]])