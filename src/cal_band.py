import copy
import numpy as np

class BandStructure():
    def __init__(self, inputs):
        self.inputs = inputs        # user input
        self.ax = self.inputs['material'].ax
        self.mesh = inputs['mesh']
    def calState(self, unit, idx):
        unit.setKx(idx)
        Heig = unit.H+\
               np.exp(1j*unit.kx*self.ax)*unit.P_plus+\
               np.exp(-1j*unit.kx*self.ax)*unit.P_minus
        val, vec = np.linalg.eig(Heig)
        return self.__sort__(val, vec)
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