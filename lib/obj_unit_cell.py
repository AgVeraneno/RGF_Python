import copy, os
import numpy as np
import lib_excel

class UnitCell():
    '''
    Unit cell object contain essential information of a unit cell
    H: Hamiltonian of the unit cell containing on site energy & interhopping.
    P_plus: forward inter unit cell hopping matrix
    P_minus: backward inter unit cell hopping matrix
    '''
    def __init__(self, inputs, u_idx):
        ## import inputs
        self.inputs = inputs                        # user inputs
        self.info = inputs['Unit cell'][u_idx]      # unit cell information
        self.mat = inputs['material']               # material object
        if self.info['Type'] == 1:                  # True if type is 2-2-2-2
            self.incTop = False
        else:
            self.incTop = True
        # Unit width, in unit of subunit cell
        self.u_wid = self.info['W']+\
                     self.info['Barrier']['top width']+\
                     self.info['Barrier']['bot width']+\
                     self.info['Type']
        # Matrix size.
        self.m_BLG = self.u_wid*8
        self.m_MLG = self.u_wid*4
        ## create default matrix
        self.H = np.zeros((self.m_BLG,self.m_BLG), dtype=np.complex128)
        self.P_plus = np.zeros((self.m_BLG,self.m_BLG), dtype=np.complex128)
        self.P_minus = np.zeros((self.m_BLG,self.m_BLG), dtype=np.complex128)
    def genHamiltonian(self):
        ## create data entrance points
        self.AB_start = self.info['Shift']              # AB layer start point
        self.AB_stop = self.AB_start+self.u_wid-1       # AB layer stop point
        self.ab_start = self.info['Shift']+self.u_wid   # ab layer start point
        self.ab_stop = self.ab_start+self.u_wid-1       # ab layer stop point
        ## generate matrix
        if self.inputs['direction'] == 'Armchair':
            self.__armchair_matrix_unit__()
            self.__ArmchairOD__()
            self.__ArmchairD__()
        ## degenerate to MLG
        if self.inputs['lattice'] == 'MLG':
            MLGstart = 0
            MLGstop = self.u_wid*4
            self.H = self.H[MLGstart:MLGstop,MLGstart:MLGstop]
            self.P_plus = self.P_plus[MLGstart:MLGstop,MLGstart:MLGstop]
            self.P_minus = self.P_minus[MLGstart:MLGstop,MLGstart:MLGstop]
    def setKx(self, l_idx):
        self.kx = 2*l_idx*np.pi/(self.mat.ax*self.inputs['kx_mesh'])
        self.kx_norm = self.kx*self.mat.ax/np.pi
    def __ArmchairD__(self):
        zone = self.info
        ## get on site energy components ##
        # calculate bias
        Vt = zone['Vtop'] - zone['Vbot']        # voltage drop across unit
        dVt = Vt/(self.u_wid*2)
        delta_ch = zone['delta']                # channel gap
        delta_bt = zone['Barrier']['top gap']   # top barrier gap
        delta_bb = zone['Barrier']['bot gap']   # bottom barrier gap
        w_bt = zone['Barrier']['top width']     # top barrier gap
        w_bb = zone['Barrier']['bot width']     # top barrier gap
        if self.inputs['lattice'] == 'MLG':
            B_inv = -1
        else:
            B_inv = 1
        for m in range(self.m_BLG):
            block = int(m/4)
            ##
            if (block >= self.AB_start and block < self.AB_start+w_bb)\
            or (block >= self.ab_start and block < self.ab_start+w_bb):
                site_delta = m%2*B_inv*delta_bb + (1-m%2)*delta_bb
                site_V1 = zone['Vbot']
                site_V2 = zone['Vbot']
            elif (block >= self.AB_stop-w_bt+1 and block <= self.AB_stop)\
            or   (block >= self.ab_stop-w_bt+1 and block <= self.ab_stop):
                site_delta = m%2*B_inv*delta_bt + (1-m%2)*delta_bt
                site_V1 = zone['Vtop']
                site_V2 = zone['Vtop']
            else:
                site_delta = m%2*B_inv*delta_ch + (1-m%2)*delta_ch
                site_V1 = (zone['Vbot']+(block-self.AB_start-w_bt)*dVt)
                site_V2 = (zone['Vbot']+(block-self.AB_start-w_bt+1)*dVt)
            ##
            if block < self.AB_start:       # shift area
                self.H[m,m] = 1000
            elif self.incTop and (block == self.AB_stop or block == self.ab_stop):# top edge
                if block == self.AB_stop:
                    if m % 4 == 0 or m % 4 == 3:# empty atoms
                        self.H[m,m] = 1000
                    else:                   # AB top edge
                        self.H[m,m] = site_delta+site_V1
                elif block == self.ab_stop:
                    if m % 4 == 1 or m % 4 == 2:# empty atoms
                        self.H[m,m] = 1000
                    else:                   # ab top edge
                        self.H[m,m] = -site_delta+site_V1
            elif block == self.AB_start:    # AB bottom edge
                if m % 4 == 1 or m % 4 == 2:
                    self.H[m,m] = 1000
                else:
                    self.H[m,m] = site_delta+site_V2
            elif block == self.ab_start:    # ab bottom edge
                if m % 4 == 0 or m % 4 == 3:
                    self.H[m,m] = 1000
                else:
                    self.H[m,m] = -site_delta+site_V2
            elif block >= self.AB_start and block < (self.AB_stop+1):
                if m % 4 == 1 or m % 4 == 2:        # AB lower atoms
                    self.H[m,m] = site_delta+site_V1
                elif m % 4 == 0 or m % 4 == 3:      # AB higher atoms
                    self.H[m,m] = site_delta+site_V2
            elif block >= self.ab_start and block < (self.ab_stop+1):
                if m % 4 == 0 or m % 4 == 3:        # ab lower atoms
                    self.H[m,m] = -site_delta+site_V1
                elif m % 4 == 1 or m % 4 == 2:      # ab higher atoms
                    self.H[m,m] = -site_delta+site_V2
    def __ArmchairOD__(self):
        '''
        Off diagonal term
        '''
        ## unit cell H
        np_matrix = []
        np_matrixP = []
        for r in range(int(self.u_wid*2)):
            row = []
            rowP = []
            for c in range(int(self.u_wid*2)):
                if r == self.AB_start or r == self.ab_start:
                    ## Bottom edge ##
                    if r == self.AB_start:
                        ## AB layer bottom ##
                        if r == c:                  # on site energy
                            row.append(self.__AB2AB_bot__)
                            rowP.append(np.zeros((4,4), dtype=np.complex128))
                        elif r == c-1:              # inter cell hopping AB
                            row.append(self.__AB2ABnext_bot__)
                            rowP.append(np.zeros((4,4), dtype=np.complex128))
                        elif r == c-self.u_wid:  # inter layer hopping
                            row.append(self.__AB2ab_bot__)
                            rowP.append(np.zeros((4,4), dtype=np.complex128))
                        elif r == c-self.u_wid-1:# inter layer AB to next ab
                            row.append(self.__AB2abnext_bot__)
                            rowP.append(np.zeros((4,4), dtype=np.complex128))
                        else:
                            row.append(np.zeros((4,4), dtype=np.complex128))
                            rowP.append(np.zeros((4,4), dtype=np.complex128))
                    elif r == self.ab_start:
                        ## ab layer bottom ##
                        if r == c:                  # on site energy
                            row.append(self.__ab2ab_bot__)
                            rowP.append(self.__ab2ab_P__)
                        elif r == c-1:              # inter cell hopping ab
                            row.append(self.__ab2abnext_bot__)
                            rowP.append(np.zeros((4,4), dtype=np.complex128))
                        else:
                            row.append(np.zeros((4,4), dtype=np.complex128))
                            rowP.append(np.zeros((4,4), dtype=np.complex128))
                    else:
                        row.append(np.zeros((4,4), dtype=np.complex128))
                        rowP.append(np.zeros((4,4), dtype=np.complex128))
                elif (r == self.AB_stop or r == self.ab_stop) and self.incTop:
                    ## Top edge ##
                    if r == self.AB_stop:
                        ## AB layer top ##
                        if r == c:                  # on site energy
                            row.append(self.__AB2AB_top__)
                            rowP.append(self.__AB2AB_P_top__)
                        elif r == c-self.u_wid:      # inter layer hopping
                            row.append(self.__AB2ab_top__)
                            rowP.append(np.zeros((4,4), dtype=np.complex128))
                        elif r == c-self.u_wid+1:    # inter layer AB to ab pre
                            row.append(np.zeros((4,4), dtype=np.complex128))
                            rowP.append(self.__AB2abpre_P__)
                        else:
                            row.append(np.zeros((4,4), dtype=np.complex128))
                            rowP.append(np.zeros((4,4), dtype=np.complex128))
                    elif r == self.ab_stop:
                        if r == c:                  # on site energy
                            row.append(self.__ab2ab_top__)
                            rowP.append(np.zeros((4,4), dtype=np.complex128))
                        else:
                            row.append(np.zeros((4,4), dtype=np.complex128))
                            rowP.append(np.zeros((4,4), dtype=np.complex128))
                    else:
                        row.append(np.zeros((4,4), dtype=np.complex128))
                        rowP.append(np.zeros((4,4), dtype=np.complex128))
                elif r > self.AB_start and r <= self.AB_stop:
                    ## AB layer cell ##
                    if r == c:                      # on site energy AB layer
                        row.append(self.__AB2AB__)
                        rowP.append(self.__AB2AB_P__)
                    elif r == c-self.u_wid:          # inter layer hopping
                        row.append(self.__AB2ab__)
                        rowP.append(self.__AB2ab_P__)
                    elif not r == self.AB_stop:
                        if r == c-1:                # AB inter sub unit cell hopping
                            row.append(self.__AB2ABnext__)
                            rowP.append(np.zeros((4,4), dtype=np.complex128))
                        elif r == c-self.u_wid-1:    # inter layer AB to next ab
                            row.append(self.__AB2abnext__)
                            rowP.append(np.zeros((4,4), dtype=np.complex128))
                        elif r == c-self.u_wid+1:
                            row.append(np.zeros((4,4), dtype=np.complex128))
                            rowP.append(self.__AB2abpre_P__)
                        else:
                            row.append(np.zeros((4,4), dtype=np.complex128))
                            rowP.append(np.zeros((4,4), dtype=np.complex128))
                    else:
                        row.append(np.zeros((4,4), dtype=np.complex128))
                        rowP.append(np.zeros((4,4), dtype=np.complex128))
                elif r > self.ab_start and r <= self.ab_stop:
                    ## ab layer cell ##
                    if r == c:                      # on site energy ab layer
                        row.append(self.__ab2ab__)
                        rowP.append(self.__ab2ab_P__)
                    elif not r == self.ab_stop:
                        if r == c-1:
                            row.append(self.__ab2abnext__)
                            rowP.append(np.zeros((4,4), dtype=np.complex128))
                        else:
                            row.append(np.zeros((4,4), dtype=np.complex128))
                            rowP.append(np.zeros((4,4), dtype=np.complex128))
                    else:
                        row.append(np.zeros((4,4), dtype=np.complex128))
                        rowP.append(np.zeros((4,4), dtype=np.complex128))
                else:
                    row.append(np.zeros((4,4), dtype=np.complex128))
                    rowP.append(np.zeros((4,4), dtype=np.complex128))
            np_matrix.append(row)
            np_matrixP.append(rowP)
        self.H = np.block(np_matrix)
        self.H = self.H + np.transpose(np.conj(self.H))
        self.P_plus = np.block(np_matrixP)
        self.P_minus = np.transpose(np.conj(self.P_plus))
    def __armchair_matrix_unit__(self):
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
        # within sub unit cell hopping
        self.__AB2AB__ = np.zeros((4,4), dtype=np.complex128)
        self.__ab2ab__ = np.zeros((4,4), dtype=np.complex128)
        self.__AB2AB_bot__ = np.zeros((4,4), dtype=np.complex128)
        self.__ab2ab_bot__ = np.zeros((4,4), dtype=np.complex128)
        self.__AB2AB_top__ = np.zeros((4,4), dtype=np.complex128)
        self.__ab2ab_top__ = np.zeros((4,4), dtype=np.complex128)
        # next sub unit cell hopping
        self.__AB2ABnext__ = np.zeros((4,4), dtype=np.complex128)
        self.__ab2abnext__ = np.zeros((4,4), dtype=np.complex128)
        self.__AB2ABnext_bot__ = np.zeros((4,4), dtype=np.complex128)
        self.__ab2abnext_bot__ = np.zeros((4,4), dtype=np.complex128)
        self.__AB2ABnext_top__ = np.zeros((4,4), dtype=np.complex128)
        self.__ab2abnext_top__ = np.zeros((4,4), dtype=np.complex128)
        # within sub unit cell inter layer hopping
        self.__AB2ab__ = np.zeros((4,4), dtype=np.complex128)
        self.__AB2ab_top__ = np.zeros((4,4), dtype=np.complex128)
        self.__AB2ab_bot__ = np.zeros((4,4), dtype=np.complex128)
        # next sub unit cell inter layer hopping
        self.__AB2abnext__ = np.zeros((4,4), dtype=np.complex128)
        self.__AB2abpre__ = np.zeros((4,4), dtype=np.complex128)
        self.__AB2abnext_top__ = np.zeros((4,4), dtype=np.complex128)
        self.__AB2abpre_top__ = np.zeros((4,4), dtype=np.complex128)
        self.__AB2abnext_bot__ = np.zeros((4,4), dtype=np.complex128)
        self.__AB2abpre_bot__ = np.zeros((4,4), dtype=np.complex128)
        # next unit cell, same sub unit cell, same layer hopping
        self.__AB2AB_P__ = np.zeros((4,4), dtype=np.complex128)
        self.__ab2ab_P__ = np.zeros((4,4), dtype=np.complex128)
        self.__AB2AB_P_top__ = np.zeros((4,4), dtype=np.complex128)
        self.__ab2ab_P_top__ = np.zeros((4,4), dtype=np.complex128)
        self.__AB2AB_P_bot__ = np.zeros((4,4), dtype=np.complex128)
        self.__ab2ab_P_bot__ = np.zeros((4,4), dtype=np.complex128)
        # next unit cell, same sub unit cell, different layer hopping
        self.__AB2ab_P__ = np.zeros((4,4), dtype=np.complex128)
        self.__AB2ab_P_top__ = np.zeros((4,4), dtype=np.complex128)
        self.__AB2ab_P_bot__ = np.zeros((4,4), dtype=np.complex128)
        # next unit cell, next sub unit cell, different layer hopping
        self.__AB2abnext_P__ = np.zeros((4,4), dtype=np.complex128)
        self.__AB2abpre_P__ = np.zeros((4,4), dtype=np.complex128)
        self.__AB2abnext_P_top__ = np.zeros((4,4), dtype=np.complex128)
        self.__AB2abpre_P_top__ = np.zeros((4,4), dtype=np.complex128)
        self.__AB2abnext_P_bot__ = np.zeros((4,4), dtype=np.complex128)
        self.__AB2abpre_P_bot__ = np.zeros((4,4), dtype=np.complex128)
        '''
        AB/ab intra and inter layer hopping
              Hsu                 P+                 AB->ab               P+ AB->ab
        a   b   a'  b'
        ==============       ==============       ==============       ==============
        A   B   A'  B'
        ==============       ==============       ==============       ==============
        v   r0  0   0        0   0   0   0        0   r3  0   r3       0   0   0   0
        -   v   r0  0        0   0   0   0        0   0   r1  0        0   0   0   0
        -   -   v   r0       0   0   0   0        0   0   0   r3       0   r3  0   0
        -   -   -   v        r0  0   0   0        0   0   0   0        r1  0   0   0
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
        # P+
        self.__AB2AB_P__[3,0] = -self.mat.r0
        self.__ab2ab_P__[3,0] = -self.mat.r0
        self.__AB2ab_P__[2,1] = -self.mat.r3
        self.__AB2ab_P__[3,0] = -self.mat.r1
        '''
        AB/ab intra and inter subcell hopping
           Hsu next             P+ next             AB->ab next        P+ AB->ab next
        a   b   a'  b'
        ==============       ==============       ==============       ==============
        A   B   A'  B'
        ==============       ==============       ==============       ==============
        0   r0  0   0        0   0   0   0        0   0   0   0        0   0   0   0
        0   0   0   0        0   0   0   0        0   0   0   0        0   0   0   0
        0   0   0   0        0   0   0   0        0   0   0   0        0   0   0   0
        0   0   r0  0        0   0   0   0        0   0   0   0        0   0   0   0
        ==============
        '''
        # H
        self.__AB2ABnext__[0,1] = -self.mat.r0
        self.__AB2ABnext__[3,2] = -self.mat.r0
        self.__ab2abnext__[0,1] = -self.mat.r0
        self.__ab2abnext__[3,2] = -self.mat.r0
        '''
        AB/ab intra and inter subcell hopping
           Hsu pre              P+ pre              AB->ab pre          P+ AB->ab pre
        a   b   a'  b'
        ==============       ==============       ==============       ==============
        A   B   A'  B'
        ==============       ==============       ==============       ==============
        0   0   0   0        0   0   0   0        0   0   0   0        0   0   0   0
        0   0   0   0        0   0   0   0        0   0   0   0        0   0   0   0
        0   0   0   0        0   0   0   0        0   0   0   r3       0   0   0   0
        0   0   0   0        0   0   0   0        0   0   0   0        0   0   0   0
        ==============
        '''
        # H
        self.__AB2abpre__[2,3] = -self.mat.r3
        '''
        Intra layer hopping
        '''
        self.__AB2ab__[0,1] = -self.mat.r3
        self.__AB2ab__[0,3] = -self.mat.r3
        self.__AB2ab__[1,0] = -self.mat.r1
        self.__AB2ab__[2,3] = -self.mat.r3
        self.__AB2ab__[3,2] = -self.mat.r1
        self.__AB2ab_top__[1,0] = -self.mat.r1
        self.__AB2ab_top__[2,3] = -self.mat.r3
        self.__AB2ab_bot__[0,1] = -self.mat.r3
        self.__AB2ab_bot__[3,2] = -self.mat.r1
        self.__AB2ab_P__[2,1] = -self.mat.r3
        self.__AB2ab_P_bot__[2,1] = -self.mat.r3
        '''
        Inter layer hopping
        '''
        self.__AB2abnext__[0,3] = -self.mat.r3
        self.__AB2abnext_bot__[0,3] = -self.mat.r3
        self.__AB2abpre_P__[2,1] = -self.mat.r3
        self.__AB2abpre_P_top__[2,1] = -self.mat.r3
