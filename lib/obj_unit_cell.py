import copy, os
import numpy as np
import lib_excel

class UnitCell():
    def __init__(self, inputs):
        ## import inputs
        self.inputs = inputs                # user inputs
        self.mat = inputs['material']       # material object
        self.wmax = self.inputs['mesh'][0]  # number of sub unit cell. Maximum ribbon width
        self.m_size = self.wmax*8           # matrix size (BLG size)
        ## create default matrix
        self.H = np.zeros((self.m_size,self.m_size), dtype=np.complex128)
        self.P_plus = np.zeros((self.m_size,self.m_size), dtype=np.complex128)
        self.P_minus = np.zeros((self.m_size,self.m_size), dtype=np.complex128)
    def genHamiltonian(self, unit):
        ## create pointers
        if unit['Type'] == 1:           # MLG Hamiltonian matrix size. 2-1-2-1 type
            self.incTop = False
            self.unit_mesh = unit['W']+1
        elif unit['Type'] == 2:         # MLG Hamiltonian matrix size. 2-2-2-2 type
            self.incTop = True
            self.unit_mesh = unit['W']+2
        self.unit_w = self.unit_mesh*4
        self.AB_start = unit['Shift']
        self.AB_stop = self.AB_start+self.unit_mesh - 1
        self.ab_start = self.wmax + unit['Shift']
        self.ab_stop = self.ab_start+self.unit_mesh - 1
        ## generate matrix
        if self.inputs['direction'] == 'Armchair':
            self.__armchair_matrix_unit__(unit)
            self.__ArmchairOD__()
            self.__ArmchairD__(unit)
        ## degenerate to MLG
        if self.inputs['lattice'] == 'MLG':
            MLGstart = 0
            MLGstop = self.wmax*4
            self.H = self.H[MLGstart:MLGstop,MLGstart:MLGstop]
            self.P_plus = self.P_plus[MLGstart:MLGstop,MLGstart:MLGstop]
            self.P_minus = self.P_minus[MLGstart:MLGstop,MLGstart:MLGstop]
    def setKx(self, len_idx):
        self.kx = 2*np.pi/(self.mat.ax*(self.inputs['mesh'][1]-1))*len_idx
        self.kx_norm = self.kx*self.mat.ax/np.pi
    def saveAsXLS(self, thisUnit):
        if self.inputs['material'].name == 'Graphene' \
        and self.inputs['direction'] == 'Armchair':
            filename = self.inputs['lattice']+'_AGNR_'
        elif self.inputs['material'].name == 'Graphene' \
        and self.inputs['direction'] == 'Zigzag':
            filename = self.inputs['lattice']+'_ZGNR_'
        condition = 'Z='+str(thisUnit['Region'])+\
                    ',Type='+str(thisUnit['Type'])+\
                    ',S='+str(thisUnit['Shift'])+\
                    ',W='+str(thisUnit['W'])+\
                    ',L='+str(thisUnit['L'])+\
                    ',Vtop='+str(thisUnit['Vtop'])+\
                    ',Vbot='+str(thisUnit['Vbot'])+\
                    ',d='+str(thisUnit['delta'])
        excel_parser = lib_excel.excel('matrix/'+filename+condition+'.xlsx')
        ## create H sheet ##
        excel_parser.newWorkbook('H')
        for i in range(np.size(self.H, 0)):
            for j in range(np.size(self.H, 1)):
                _ = excel_parser.worksheet.cell(column=j+1, row=i+1,\
                                                value="=COMPLEX("+str(np.real(self.H[i,j]))\
                                                +","+str(np.imag(self.H[i,j]))+")")
        ## create P+ sheet ##
        excel_parser.newSheet('P')
        P = self.P_plus+self.P_minus
        for i in range(np.size(self.H, 0)):
            for j in range(np.size(self.H, 1)):
                _ = excel_parser.worksheet.cell(column=j+1, row=i+1,\
                                                value="=COMPLEX("+str(np.real(P[i,j]))\
                                                +","+str(np.imag(P[i,j]))+")")
        try:
            excel_parser.save()
        except:
            os.mkdir('matrix')
            excel_parser.save()
    def __ArmchairD__(self, zone):
        Vt = zone['Vtop'] - zone['Vbot']
        dVt = Vt/(self.unit_mesh*2)
        delta = zone['delta']*1.6e-19
        if self.inputs['lattice'] == 'MLG':
            B_inv = -1
        else:
            B_inv = 1
        for m in range(self.m_size):
            block = int(m/4)
            site_V1 = (zone['Vbot']+block*dVt)*1.6e-19
            site_V2 = (zone['Vbot']+(block+1)*dVt)*1.6e-19
            site_delta = m%2*B_inv*delta + (1-m%2)*delta
            if block < self.AB_start:       # shift area
                self.H[m,m] = 0
            elif self.incTop and \
            (block == self.AB_stop or \
             block == self.ab_stop):               # top edge
                if block == self.AB_stop:
                    if m % 4 == 0 or m % 4 == 3:# empty atoms
                        self.H[m,m] = 0
                    else:                   # AB top edge
                        self.H[m,m] = site_delta+site_V1
                elif block == self.ab_stop:
                    if m % 4 == 1 or m % 4 == 2:# empty atoms
                        self.H[m,m] = 0
                    else:                   # ab top edge
                        self.H[m,m] = -site_delta+site_V1
            elif block == self.AB_start:    # AB bottom edge
                if m % 4 == 1 or m % 4 == 2:
                    self.H[m,m] = 0
                else:
                    self.H[m,m] = site_delta+site_V2
            elif block == self.ab_start:    # ab bottom edge
                if m % 4 == 0 or m % 4 == 3:
                    self.H[m,m] = 0
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
        for r in range(int(self.wmax*2)):
            row = []
            rowP = []
            for c in range(int(self.wmax*2)):
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
                        elif r == c-self.wmax:  # inter layer hopping
                            row.append(self.__AB2ab_bot__)
                            rowP.append(np.zeros((4,4), dtype=np.complex128))
                        elif r == c-self.wmax-1:# inter layer AB to next ab
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
                        elif r == c-self.wmax:      # inter layer hopping
                            row.append(self.__AB2ab_top__)
                            rowP.append(np.zeros((4,4), dtype=np.complex128))
                        elif r == c-self.wmax+1:    # inter layer AB to ab pre
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
                    elif r == c-self.wmax:          # inter layer hopping
                        row.append(self.__AB2ab__)
                        rowP.append(self.__AB2ab_P__)
                    elif not r == self.AB_stop:
                        if r == c-1:                # AB inter sub unit cell hopping
                            row.append(self.__AB2ABnext__)
                            rowP.append(np.zeros((4,4), dtype=np.complex128))
                        elif r == c-self.wmax-1:    # inter layer AB to next ab
                            row.append(self.__AB2abnext__)
                            rowP.append(np.zeros((4,4), dtype=np.complex128))
                        elif r == c-self.wmax+1:
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
    def __armchair_matrix_unit__(self, unit):
        '''
          subunit cell        bottom edge         top edge
        b    A  a',B'
        X    O-----0
            /       \
           0    X    O       X    O-----0        0    X    O
          a,B   b'   A'      b    A  a',B'      a,B   b'   A'
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
        Intra cell hopping
        '''
        self.__AB2AB__[0,1] = -self.mat.r0
        self.__AB2AB__[0,3] = -self.mat.r0
        self.__AB2AB__[2,3] = -self.mat.r0
        self.__AB2AB_bot__[0,3] = -self.mat.r0
        self.__ab2ab__[0,1] = -self.mat.r0
        self.__ab2ab__[0,3] = -self.mat.r0
        self.__ab2ab__[2,3] = -self.mat.r0
        self.__ab2ab_top__[0,3] = -self.mat.r0
        self.__AB2AB_P__[1,2] = -self.mat.r0
        self.__AB2AB_P_top__[1,2] = -self.mat.r0
        self.__ab2ab_P__[1,2] = -self.mat.r0
        self.__ab2ab_P_bot__[1,2] = -self.mat.r0
        '''
        Inter cell hopping
        '''
        self.__AB2ABnext__[0,1] = -self.mat.r0
        self.__AB2ABnext__[3,2] = -self.mat.r0
        self.__AB2ABnext_bot__[0,1] = -self.mat.r0
        self.__AB2ABnext_bot__[3,2] = -self.mat.r0
        self.__ab2abnext__[1,0] = -self.mat.r0
        self.__ab2abnext__[2,3] = -self.mat.r0
        self.__ab2abnext_bot__[1,0] = -self.mat.r0
        self.__ab2abnext_bot__[2,3] = -self.mat.r0
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
