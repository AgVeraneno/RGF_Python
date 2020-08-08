import os, copy
import numpy as np

class CPU():
    def __init__(self, setup, unit):
        self.unit = unit
        if setup['Direction'] == 'AC': self.ax = setup['Material'].ax
        elif setup['Direction'] == 'ZZ': self.ax = setup['Material'].ax_zz
        self.mat = setup['Material']
        self.direction = setup['Direction']
        self.mesh = int(setup['mesh'])
        self.lattice = setup['Lattice']
        self.rawdata = None
    def setKx(self, l_idx):
        return 2*np.pi*(l_idx-(self.mesh-1)/2)/(self.ax*(self.mesh-1))
        #return 2*np.pi*l_idx/(self.ax*(self.mesh-1))
    def calState(self, l_idx, returnKx=False):
        kx = self.setKx(l_idx)
        H = self.unit.H*self.mat.q
        Pf = self.unit.Pf*self.mat.q
        Pb = self.unit.Pb*self.mat.q
        Heig = H+\
               np.exp(-1j*kx*self.ax)*Pf+\
               np.exp(1j*kx*self.ax)*Pb
        val, vec = np.linalg.eig(Heig)
        weight = self.calWeight(vec)
        return kx, val, vec, weight
    def calWeight(self,vec):
        weight = copy.deepcopy(vec)
        for i in range(np.size(vec,0)): weight[:,i] = np.real(np.square(np.absolute(vec[:,i])))
        return weight
    def calMagneticMoment(self, kx, vec1, vec2):
        uH = self.unit.uH
        uPf = self.unit.uPf
        uPb = self.unit.uPb
        uHeig = uH+\
                np.exp(-1j*kx*self.ax)*uPf+\
                np.exp(1j*kx*self.ax)*uPb
        euH = np.vdot(vec1,np.dot(uHeig,vec2))
        uH0 = self.unit.uH0
        uPf0 = self.unit.uPf0
        uPb0 = self.unit.uPb0
        uHeig0 = uH0+\
                 np.exp(-1j*kx*self.ax)*uPf0+\
                 np.exp(1j*kx*self.ax)*uPb0
        euH0 = np.vdot(vec1,np.dot(uHeig0,vec2))
        eY = np.vdot(vec1,np.dot(self.unit.Y,vec2))
        return (euH - euH0*eY)/self.mat.uB
    def calMagneticMomentCurrent(self, vec):
        Area = 1.5*3**0.5*self.mat.acc**2
        if self.direction == 'ZZ':
            ## calculate link current
            I_link = []
            for i in range(0,len(vec),2):
                if i%2 == 0:
                    II = -1j/self.mat.h_bar*(np.dot(vec[i],np.conj(vec[i+1]))*self.mat.r0*self.mat.q - \
                                            np.dot(np.conj(vec[i]),vec[i+1])*self.mat.r0*self.mat.q)
                else:
                    II = -1j/self.mat.h_bar*(np.dot(vec[i+1],np.conj(vec[i]))*self.mat.r0*self.mat.q - \
                                            np.dot(np.conj(vec[i+1]),vec[i])*self.mat.r0*self.mat.q)
                I_link.append(II*self.mat.q*Area/self.mat.uB)
            else:
                I_link_tot =sum(I_link)
                I_trans = []
                for i in range(0,len(vec),2):
                    I_trans.append(I_link_tot*(abs(vec[i])**2+abs(vec[i+1])**2))
                I_loop = []
                for j in range(len(I_trans)):
                    if j == 0: III = I_link[j]-I_trans[j]
                    else: III = I_link[j]-I_trans[j] + I_loop[j-1]
                    I_loop.append(np.real(III))
                else:
                    return I_loop
        elif self.direction == 'AC':
            ## calculate link current
            I_link = []
            for i in range(0,len(vec),2):
                if i%2 == 0:
                    II = -1j/self.mat.h_bar*(np.dot(vec[i],np.conj(vec[i+1]))*self.mat.r0*self.mat.q - \
                                            np.dot(np.conj(vec[i]),vec[i+1])*self.mat.r0*self.mat.q)
                else:
                    II = -1j/self.mat.h_bar*(np.dot(vec[i+1],np.conj(vec[i]))*self.mat.r0*self.mat.q - \
                                            np.dot(np.conj(vec[i+1]),vec[i])*self.mat.r0*self.mat.q)
                I_link.append(II*self.mat.q*Area/self.mat.uB)
            else:
                I_link_tot =sum(I_link)
                I_trans = []
                for i in range(0,len(vec),2):
                    I_trans.append(I_link_tot*(abs(vec[i])**2+abs(vec[i+1])**2))
                I_loop = []
                for j in range(len(I_trans)):
                    if j == 0: III = I_link[j]-I_trans[j]
                    else: III = I_link[j]-I_trans[j] + I_loop[j-1]
                    I_loop.append(np.real(III))
                else:
                    return I_loop
    def getCBidx(self, gap, eig_val):
        #return int(np.size(self.unit.H,0)/2)
        return int(self.CB_idx)
        '''
        for v_idx, v in enumerate(eig_val):
            if v >= 0:
                return v_idx
        '''
    def sort_eigenstate(self, val, vec, weight=[], ref_val=[], ref_vec=[], ref_weight=[]):
        '''
        What: Sort eigenstate with small to large sequence
        How: 1.Sweep original eigenvalue and match sorted one.
             2.Copy the original eigenstate to a new array.
        inputs:
        val: eigenvalue [n*n]
        vec: eigenstate [n*n]
        '''
        if np.size(ref_val) == 0:
            sorted_val = np.sort(val)
            sorted_vec = copy.deepcopy(vec)
            ## Sort with eigenvalue
            for v1_idx, v1 in enumerate(val):
                for v2_idx, v2 in enumerate(sorted_val):
                    if v1 == v2:
                        sorted_vec[:,v2_idx] = copy.deepcopy(vec[:,v1_idx])
                    else: continue
            else:
                if np.size(weight) > 0: return sorted_val, sorted_vec, weight
                else: return sorted_val, sorted_vec
        else: ## with reference. Sort with weight
            sorted_val = copy.deepcopy(val)
            sorted_vec = copy.deepcopy(vec)
            sorted_wgt = copy.deepcopy(weight)
            for v1_idx in range(len(val)):
                dif_weight = [np.sum(np.absolute(ref_weight[:,v1_idx] - weight[:,v2_idx])) for v2_idx in range(len(val))]
                idx = dif_weight.index(min(dif_weight))
                sorted_val[v1_idx] = val[idx]
                sorted_vec[:,v1_idx] = vec[:,idx]
                sorted_wgt[:,v1_idx] = weight[:,idx]
            else:
                return sorted_val, sorted_vec, sorted_wgt
