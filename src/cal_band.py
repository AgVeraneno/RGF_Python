import os, copy
import numpy as np

class CPU():
    def __init__(self, inputs, unit):
        self.inputs = inputs        # user input
        self.unit = unit
        self.ax = self.inputs['material'].ax
        self.mesh = inputs['mesh']
        self.x = []
        self.y = []
    def calState(self, idx):
        self.unit.setKx(idx)
        kx = self.unit.kx
        H = self.unit.H
        Pp = self.unit.P_plus
        Pn = self.unit.P_minus
        Heig = H+\
               np.exp(1j*kx*self.ax)*Pp+\
               np.exp(-1j*kx*self.ax)*Pn
        val, vec = np.linalg.eig(Heig)
        return self.__sort__(val, vec)
    def calState_MP(self, idx):
        self.unit.setKx(idx)
        kx = self.unit.kx
        H = self.unit.H
        Pp = self.unit.P_plus
        Pn = self.unit.P_minus
        Heig = H+\
               np.exp(1j*kx*self.ax)*Pp+\
               np.exp(-1j*kx*self.ax)*Pn
        val, vec = np.linalg.eig(Heig)
        sort_val, sort_vec = self.__sort__(val, vec)
        return [sort_val, self.unit.kx_norm]
    def getCBidx(self, gap, eig_val):
        for v_idx, v in enumerate(eig_val):
            if v > 0 and gap - v <= 1e-4:
                return v_idx
    def __sort__(self, val, vec, mode='CPU'):
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