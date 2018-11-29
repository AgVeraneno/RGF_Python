import os, copy
import numpy as np
from matplotlib import pyplot

class BandStructure():
    def __init__(self, inputs):
        self.inputs = inputs        # user input
        if inputs['GPU']['enable']:
            global cp
            import cupy as cp
    def calState(self, unit):
        mat = self.inputs['material']
        H = unit.H+\
            np.exp(1j*unit.kx*mat.ax)*unit.P_plus+\
            np.exp(-1j*unit.kx*mat.ax)*unit.P_minus
        val, vec = np.linalg.eig(H)
        return self.__sort__(val, vec)
    def calState_GPU(self, unit):
        mat = self.inputs['material']
        H = cp.asarray(unit.H)+\
            cp.exp(1j*unit.kx*mat.ax)*cp.asarray(unit.P_plus)+\
            cp.exp(-1j*unit.kx*mat.ax)*cp.asarray(unit.P_minus)
        val, vec = cp.linalg.eigh(H)
        return self.__sort__(val, vec, 'GPU')
    def plotBand(self, bandgap_list, unit_idx):
        ## construct vectors
        eig_mat = np.real(np.array(bandgap_list['y']))
        kx_sweep = np.real(np.array(bandgap_list['x']))
        for y_idx in range(np.size(eig_mat,1)):
            pyplot.plot(kx_sweep,eig_mat[:,y_idx]/1.6e-19)
            pyplot.xlim([0,1])
            pyplot.xlabel("kx*ax/pi")
            pyplot.ylabel("E (eV)")
        ## output to figures
        thisUnit = self.inputs['Unit cell'][unit_idx]
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
        try:
            pyplot.savefig('figures/'+filename+condition+'.png')
        except:
            os.mkdir('figures')
            pyplot.savefig('figures/'+filename+condition+'.png')
        pyplot.close()
        ## plot zoom in figure if enabled
        if self.inputs['function']['isPlotZoom']:
            for y_idx in range(np.size(eig_mat,1)):
                pyplot.plot(kx_sweep,eig_mat[:,y_idx]/1.6e-19)
                pyplot.xlim([0,1])
                pyplot.ylim([-1,1])
                pyplot.xlabel("kx*ax/pi")
                pyplot.ylabel("E (eV)")
            try:
                pyplot.savefig('figures/'+filename+condition+'_zoom.png')
            except:
                os.mkdir('figures')
                pyplot.savefig('figures/'+filename+condition+'_zoom.png')
            pyplot.close()
    def __sort__(self, val, vec, caltype='CPU'):
        """
        What: Sort eigenstate with small to large sequence
        How: 1.Sweep original eigenvalue and match sorted one.
             2.Copy the original eigenstate to a new array. 
        """
        if caltype == 'GPU':
            output_vec = cp.zeros((cp.size(vec,0),cp.size(vec,0)), dtype=cp.complex128)
            sorted_val = cp.sort(val)
        else:
            output_vec = np.zeros((np.size(vec,0),np.size(vec,0)), dtype=np.complex128)
            sorted_val = np.sort(val)
        for v1_idx, v1 in enumerate(val):
            for v2_idx, v2 in enumerate(sorted_val):
                if v1 == v2:
                    output_vec[:,v2_idx] = copy.deepcopy(vec[:, v1_idx])
                    break
        return sorted_val, output_vec