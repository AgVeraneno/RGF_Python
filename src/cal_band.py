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
    def setKx(self, l_idx):
        return 2*np.pi*(l_idx-(self.mesh-1)/2)/(self.ax*(self.mesh-1))
    def calState(self, l_idx, returnKx=False):
        kx = self.setKx(l_idx)
        H = self.unit.H*self.mat.q
        Pf = self.unit.Pf*self.mat.q
        Pb = self.unit.Pb*self.mat.q
        Heig = H+\
               np.exp(-1j*kx*self.ax)*Pf+\
               np.exp(1j*kx*self.ax)*Pb
        val, vec = np.linalg.eig(Heig)
        return kx, val, vec
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
    def sort_eigenstate_bak(self, val, vec, ref_vec=[]):
        sorted_val = np.sort(val)
        sorted_vec = copy.deepcopy(vec)
        if not len(ref_vec)==0:     # match with previous data (beta)
            ref_vec_head = ref_vec[0,:]
            vec_head = vec[0,:]
            for v1_idx in range(len(val)):
                for v2_idx in range(len(val)):
                    if abs(sum(vec[:,v1_idx] - ref_vec[:,v2_idx])) < 1e-6:
                        sorted_val[v2_idx] = val[v1_idx]
                        sorted_vec[:,v2_idx] = vec[:,v1_idx]
                        
        else:                   # auto sort from small to large
            for v1_idx, v1 in enumerate(val):
                for v2_idx, v2 in enumerate(sorted_val):
                    if v1 == v2: sorted_vec[:,v2_idx] = copy.deepcopy(vec[:,v1_idx])
            else:
                return sorted_val, sorted_vec
    def sort_eigenstate(self, val, vec):
        sorted_val = np.sort(val)
        sorted_vec = copy.deepcopy(vec)                 # auto sort from small to large
        for v1_idx, v1 in enumerate(val):
            for v2_idx, v2 in enumerate(sorted_val):
                if v1 == v2: sorted_vec[:,v2_idx] = copy.deepcopy(vec[:,v1_idx])
        else:
            return sorted_val, sorted_vec
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
