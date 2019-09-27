import os, copy
import numpy as np

class CPU():
    def __init__(self, setup, unit):
        self.unit = unit
        self.ax = setup['material'].ax
        self.mesh = int(setup['kx_mesh'])
        self.val = []
    def setKx(self, l_idx):
        return 2*np.pi*l_idx/(self.ax*(self.mesh-1))
    def calState(self, l_idx, returnKx=False):
        kx = self.setKx(l_idx)
        H = self.unit.H
        Pf = self.unit.Pf
        Pb = self.unit.Pb
        Heig = H+\
               np.exp(1j*kx*self.ax)*Pb+\
               np.exp(-1j*kx*self.ax)*Pf
        val, vec = np.linalg.eig(Heig)
        val, vec = self.__sort__(val, vec)
        if returnKx:
            return kx, val, vec
        else:
            return kx*self.ax/np.pi, val, vec
    def getCBidx(self, gap, eig_val):
        #return int(np.size(self.unit.H,0)/2)
        return int(self.CB_idx)
        '''
        for v_idx, v in enumerate(eig_val):
            if v >= 0:
                return v_idx
        '''
    def __sort__(self, val, vec):
        """
        What: Sort eigenstate with small to large sequence
        How: 1.Sweep original eigenvalue and match sorted one.
             2.Copy the original eigenstate to a new array.
        inputs:
        val: eigenvalue [n*n]
        vec: eigenstate [n*n]
        """
        vec_size = np.size(vec,0)
        output_vec = np.zeros((vec_size,vec_size), dtype=np.complex128)
        sorted_val = np.sort(val)
        for v1_idx, v1 in enumerate(val):
            for v2_idx, v2 in enumerate(sorted_val):
                if v1 == v2:
                    output_vec[:,v2_idx] = copy.deepcopy(vec[:, v1_idx])
                    break
        return sorted_val, output_vec