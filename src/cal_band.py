import os, copy
import numpy as np
import IO_util

class CPU():
    def __init__(self, setup, unit):
        self.unit = unit
        if setup['Direction'] == 'AC': self.ax = setup['Material'].ax
        elif setup['Direction'] == 'ZZ': self.ax = setup['Material'].ax_zz
        self.a = setup['Material'].a
        self.mat = setup['Material']
        self.direction = setup['Direction']
        self.mesh = int(setup['mesh'])
        self.lattice = setup['Lattice']
        self.rawdata = None
    def setKx(self, l_idx):
        return 2*np.pi*(l_idx-(self.mesh-1)/2)/(self.ax*(self.mesh-1))
        #return 2*np.pi*l_idx/(self.ax*(self.mesh-1))
    def calState(self, l_idx):
        H = self.unit.H*self.mat.q
        Pf = self.unit.Pf*self.mat.q
        Pb = self.unit.Pb*self.mat.q
        # calculate eigenstate in kx = l_idx
        kx = self.setKx(l_idx)
        Heig = H+\
               np.exp(-1j*kx*self.ax)*Pf+\
               np.exp(1j*kx*self.ax)*Pb
        val, vec = np.linalg.eig(Heig)
        val, vec = self.__sort__(val, vec, 'energy')
        wgt = self.calWeight(vec)
        return kx, val, vec, wgt
    def calWeight(self,vec):
        weight = copy.deepcopy(vec)
        for i in range(np.size(vec,0)): weight[:,i] = np.real(np.square(np.abs(vec[:,i])))
        return weight
    def calMagneticMoment(self, kx, vec1, vec2, debug=False):
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
        uB_star = self.mat.uB/(self.mat.eff_me_ratio*self.unit.delta*self.mat.q)
        if debug: return (euH)/self.mat.uB, euH0*eY/self.mat.uB, (euH- euH0*eY)/uB_star
        else: return (euH- euH0*eY)/uB_star
        
    def calMagneticMomentCurrent(self, vec):
        Area = 1.5*3**0.5*self.mat.acc**2
        uB_star = self.mat.uB/(self.mat.eff_me_ratio*self.unit.delta*self.mat.q)
        if self.direction == 'ZZ':
            ## calculate link current
            I_link = []
            for i in range(0,len(vec),2):
                if (i/2)%2 == 0:
                    II = -1j/self.mat.h_bar*(np.dot(vec[i],np.conj(vec[i+1]))*self.mat.r0*self.mat.q - \
                                                np.dot(np.conj(vec[i]),vec[i+1])*self.mat.r0*self.mat.q)
                else:
                    II = -1j/self.mat.h_bar*(np.dot(vec[i+1],np.conj(vec[i]))*self.mat.r0*self.mat.q - \
                                                np.dot(np.conj(vec[i+1]),vec[i])*self.mat.r0*self.mat.q)
                I_link.append(II*self.mat.q*Area/uB_star)
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
            pass
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
                dif_weight = [np.sum(np.abs(np.subtract(ref_weight[:,v1_idx],weight[:,v2_idx]))) for v2_idx in range(len(val))]
                idx = dif_weight.index(min(dif_weight))
                sorted_val[v1_idx] = val[idx]
                sorted_vec[:,v1_idx] = vec[:,idx]
                sorted_wgt[:,v1_idx] = weight[:,idx]
            else:
                return sorted_val, sorted_vec, sorted_wgt
    def __sort__(self, val, vec, srt_type, wgt=None, ref_wgt=None):
        if srt_type == 'weight':
            sorted_val = copy.deepcopy(val)
            sorted_vec = copy.deepcopy(vec)
            sorted_wgt = copy.deepcopy(wgt)
            for w_idx in range(len(val)):
                dif_weight = [np.sum(np.abs(np.subtract(ref_wgt[0,w_idx],wgt[0,v2_idx]))) for v2_idx in range(len(val))]
                idx = dif_weight.index(min(dif_weight))
                sorted_val[w_idx] = val[idx]
                sorted_vec[:,w_idx] = vec[:,idx]
                sorted_wgt[:,w_idx] = wgt[:,idx]
            else: return sorted_val, sorted_vec, sorted_wgt
        elif srt_type == 'energy':
            sorted_val = np.sort(val)
            sorted_vec = copy.deepcopy(vec)
            ## Sort with eigenvalue
            for v1_idx, v1 in enumerate(val):
                for v2_idx, v2 in enumerate(sorted_val):
                    if v1 == v2:
                        sorted_vec[:,v2_idx] = copy.deepcopy(vec[:,v1_idx])
                    else: continue
            else: return sorted_val, sorted_vec
        elif srt_type == 'align':
            sorted_val = np.sort(val)
            sort_idx = []
            ## Sort with eigenvalue
            for v1_idx, v1 in enumerate(val):
                for v2_idx, v2 in enumerate(sorted_val):
                    if v1 == v2:
                        sort_idx.append(v2_idx)
                        break
                    else: continue
            else: return sort_idx
    def refreshBands(self, val, vec, srt_idx):
        sorted_val = copy.deepcopy(val)
        sorted_vec = copy.deepcopy(vec)
        for pre_idx, post_idx in enumerate(srt_idx):
            sorted_val[post_idx] = val[pre_idx]
            sorted_vec[:,post_idx] = vec[:,pre_idx]
        else: return sorted_val, sorted_vec
    def saveBand(self, rawdata, unit, folder):
        band_table = [['kx*a']]
        weight_table = [['Band','kx*a']]
        for i in range(len(rawdata[0][1])):
            band_table[0].append('Band'+str(i+1)+' (eV)')
            weight_table[0].append('Site'+str(i+1))
        for e_idx, e in enumerate(rawdata):
            kx = e[0]*self.a
            val = e[1]/1.6e-19
            vec = e[2]
            ## append data to table
            band_table.append([kx])
            band_table[-1].extend(np.real(val))
            for E_idx in unit.region['E_idx']:
                if e_idx in unit.region['S_idx']:
                    ## eigenstate weight table
                    weight_table.append([E_idx])
                    weight_table[-1].append(kx)
                    weight_table[-1].extend(abs(vec[:,E_idx-1])**2)
        else:
            ## print out report
            IO_util.saveAsCSV(folder+'_band.csv', band_table)
            IO_util.saveAsCSV(folder+'_weight.csv', weight_table)
    def saveMagneticMoment(self, rawdata, unit, folder):
        uTB = [['Band','kx*a','muTB']]
        I_loop = [['Band','kx*a','uB_tot']]
        for i in range(int(len(rawdata[0][1])/2)):
            I_loop[0].append('Hex'+str(i+1))
        for e_idx, e in enumerate(rawdata):
            for E_idx in unit.region['E_idx']:
                if e_idx in unit.region['S_idx']:
                    kx = e[0]
                    vec = e[2]
                    ## magnetic moment
                    uTB.append([])
                    uTB[-1].append(E_idx)
                    uTB[-1].append(kx*self.a)
                    uA,uB, uTB_val = self.calMagneticMoment(kx, vec[:,E_idx-1], vec[:,E_idx-1], True)
                    uTB[-1].append(np.real(uTB_val))
                    uTB[-1].append(np.real(uA))
                    uTB[-1].append(np.real(uB))
                    ## moment current
                    I_loop.append([])
                    I_loop[-1].append(E_idx)
                    I_loop[-1].append(kx*self.a)
                    I_list = self.calMagneticMomentCurrent(vec[:,E_idx-1])
                    I_loop[-1].append(sum(I_list))
                    I_loop[-1].extend(I_list)
        else:
            ## print out report
            IO_util.saveAsCSV(folder+'_uTB.csv', uTB)
            IO_util.saveAsCSV(folder+'_Iloop.csv', I_loop)