import os, copy
import numpy as np

class CPU():
    def __init__(self, setup, unit):
        self.unit = unit
        self.ax = setup['material'].ax
        self.a = setup['material'].a
        self.mesh = int(setup['kx_mesh'])
        self.SU_type = setup['SU_type']
        self.lattice = setup['lattice']
    def setKx(self, l_idx):
        return 2*np.pi*(l_idx-(self.mesh-1)/2)/(self.ax*(self.mesh-1))
    def calState(self, l_idx, returnKx=False, ref_val=None, ref_vec=None):
        kx = self.setKx(l_idx)
        H = self.unit.H
        Pf = self.unit.Pf
        Pb = self.unit.Pb
        Heig = H+\
               np.exp(1j*kx*self.ax)*Pf+\
               np.exp(-1j*kx*self.ax)*Pb
        val, vec = np.linalg.eig(Heig)
        val, vec, sorted_vec = self.__sort__(val, vec)
        if returnKx:
            return kx, val, vec, sorted_vec
        else:
            return kx*self.a, val, vec, sorted_vec
    def getCBidx(self, gap, eig_val):
        #return int(np.size(self.unit.H,0)/2)
        return int(self.CB_idx)
        '''
        for v_idx, v in enumerate(eig_val):
            if v >= 0:
                return v_idx
        '''
    def __sort__(self, val, vec, ref_vec=np.zeros(0)):
        """
        What: Sort eigenstate with small to large sequence
        How: 1.Sweep original eigenvalue and match sorted one.
             2.Copy the original eigenstate to a new array.
        inputs:
        val: eigenvalue [n*n]
        vec: eigenstate [n*n]
        """
        if np.size(ref_vec) == 0:
            vec_size = np.size(vec,0)
            output_vec = np.zeros((vec_size,vec_size), dtype=np.complex128)
            sorted_vec = np.zeros((vec_size,vec_size), dtype=np.complex128)
            sorted_val = np.sort(val)
            for v1_idx, v1 in enumerate(val):
                for v2_idx, v2 in enumerate(sorted_val):
                    if v1 == v2:
                        output_vec[:,v2_idx] = copy.deepcopy(vec[:, v1_idx])
                        if self.lattice == 'MLG':
                            sorted_vec[:,v2_idx] = self.sort_vec(vec[:, v1_idx])
                        elif self.lattice == 'BLG':
                            mid = int(len(vec[:,v1_idx])/2)
                            m_size = len(vec[:,v1_idx])
                            sorted_vec[0:mid,v2_idx] = self.sort_vec(vec[0:mid, v1_idx])
                            sorted_vec[mid:m_size,v2_idx] = self.sort_vec(vec[mid:m_size, v1_idx])
                        break
            return sorted_val, output_vec, sorted_vec
        else:
            '''
            under testing for eigenstate matching
            '''
            vec_size = np.size(vec,0)
            output_vec = np.zeros((vec_size,vec_size), dtype=np.complex128)
            sorted_vec = np.zeros((vec_size,vec_size), dtype=np.complex128)
            tmp_vec = np.zeros((vec_size,vec_size), dtype=np.complex128)
            sorted_val = np.zeros(vec_size, dtype=np.complex128)
            ## rearrange eigenstate
            for v1_idx in range(len(val)):
                if self.lattice == 'MLG':
                    tmp_vec[:,v1_idx] = self.sort_vec(vec[:, v1_idx])
                elif self.lattice == 'BLG':
                    mid = int(len(vec[:,v1_idx])/2)
                    m_size = len(vec[:,v1_idx])
                    tmp_vec[0:mid,v1_idx] = self.sort_vec(vec[0:mid, v1_idx])
                    tmp_vec[mid:m_size,v1_idx] = self.sort_vec(vec[mid:m_size, v1_idx])
            ## find minimum eigenstate difference index
            for v_idx, v in enumerate(val):
                diff = [abs(sum(np.real(np.around(ref_vec[:,i],10) - np.around(tmp_vec[:,v_idx],10)))) for i in range(len(val))]
                idx = diff.index(min(diff))
                sorted_val[idx] = v
                sorted_vec[:,idx] = tmp_vec[:,v_idx]
                output_vec[:,idx] = vec[:,v_idx]
            else:
                return sorted_val, output_vec, sorted_vec
    def sort_vec(self, vec):
        idx_list = []
        unit_count = int(len(vec)/self.unit.SU_count/self.unit.SU_size)
        sep_count = int(unit_count/2) + unit_count%2
        ovl_count = unit_count - sep_count
        for s in range(sep_count):
            for i in range(self.unit.SU_size):
                idx_list.append(2*s+i)
                idx_list.append(2*s+i+unit_count)
        for o in range(ovl_count):
            for i in range(self.unit.SU_size):
                idx_list.append(2*o+1+i)
                idx_list.append(2*o+1+unit_count+i)
        sorted_vec = copy.deepcopy(vec)
        for v_idx in range(len(vec)):
            sorted_vec[idx_list[v_idx]] = abs(vec[v_idx])**2
        else:
            return sorted_vec
