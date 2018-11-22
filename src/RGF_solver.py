import sys, os
sys.path.append('../lib/')
import numpy as np
import time

from PyQt4 import QtGui
import lib_material, lib_excel, obj_unit_cell
from matplotlib import pyplot as mplot

def importSetting(filename=None):
    inputs = {'material': None,
              'lattice': None,
              'direction': None,
              'mesh': [0, 0],
              'Vbias': [0.0, 0.0],
              'Unit cell': [],
              'GPU':{'enable': False,
                     'Max matrix':4000}}
    with lib_excel.excel(filename) as excel_parser:
        for row in excel_parser.readSheet('__setup__'):
            if row[0].value == 'Using GPU':
                inputs['GPU']['enable'] = bool(row[1].value)
            elif row[0].value == 'Material':
                if str(row[1].value) == 'Graphene':
                    inputs['material'] = lib_material.Graphene()
            elif row[0].value == 'Lattice':
                inputs['lattice'] = str(row[1].value)
            elif row[0].value == 'Direction':
                inputs['direction'] = str(row[1].value)
            elif row[0].value == 'Max ribbon width':
                inputs['mesh'][0] = int(row[1].value)
            elif row[0].value == 'Max ribbon length':
                inputs['mesh'][1] = int(row[1].value)
            elif row[0].value == 'Bias(V)':
                inputs['Vbias'][0] = float(row[1].value)
                inputs['Vbias'][1] = float(row[2].value)
            elif row[0].value == 'o':
                new_unit = {'Region': int(row[1].value),
                            'Type': int(row[2].value),
                            'Shift': int(row[3].value),
                            'W': int(row[4].value),
                            'L': int(row[5].value),
                            'Vtop': float(row[6].value),
                            'Vbot': float(row[7].value),
                            'delta': float(row[8].value)}
                inputs['Unit cell'].append(new_unit)
        if inputs['GPU']['enable']:
            global cp
            import cupy as cp
    return inputs

def generateUnitCell(inputs):
    unit_list = []
    for idx in range(len(inputs['Unit cell'])):
        new_unitcell = obj_unit_cell.UnitCell(inputs)
        new_unitcell.genHamiltonian(inputs['Unit cell'][idx])
        unit_list.append(new_unitcell)
    return unit_list

def calculateState(inputs, unit):
    mat = inputs['material']
    if inputs['GPU']['enable']:
        H_eig = cp.array(unit.H)+\
                cp.exp(1j*unit.kx*mat.ax)*cp.array(unit.P_plus)+\
                cp.exp(-1j*unit.kx*mat.ax)*cp.array(unit.P_minus)
        eigVal_cp, eigVec_cp = cp.linalg.eigh(H_eig)
        eigVal = cp.asnumpy(eigVal_cp)
        eigVec = cp.asnumpy(eigVec_cp)
    else:
        H = unit.H+\
            np.exp(1j*unit.kx*mat.ax)*unit.P_plus+\
            np.exp(-1j*unit.kx*mat.ax)*unit.P_minus
        eigVal, eigVec = np.linalg.eig(H)
    return eigVal, eigVec
def plotBandgap(bandgap_list):
    eig_mat = np.real(np.array(bandgap_list['y']))
    kx_sweep = np.real(np.array(bandgap_list['x']))
    for y_idx in range(np.size(eig_mat,1)):
        mplot.plot(kx_sweep,eig_mat[:,y_idx]/1.6e-19)
        mplot.xlim([0,1])

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    '''
    This program simulates ballistic transpotation along x-axis.
    '''
    inputs = importSetting('RGF_input_file.xlsx')
    '''
    Generate RGF unit cell
    '''
    unit_list = generateUnitCell(inputs)
    '''
    calculate band structure and incident state
    '''
    for unit_idx, unit in enumerate(unit_list):
        bandgap_list = {'x':[],'y':[]}
        for idx in range(int(inputs['Unit cell'][unit_idx]['L']-1)):
            unit.setKx(unit_idx, idx)
            val,vec = calculateState(inputs, unit)
            bandgap_list['x'].append(unit.kx_norm)
            bandgap_list['y'].append(np.sort(val))
        plotBandgap(bandgap_list)
        try:
            thisUnit = inputs['Unit cell'][unit_idx]
            if inputs['material'].name == 'Graphene' and inputs['direction'] == 'Armchair':
                filename = inputs['lattice']+'_AGNR_'
            elif inputs['material'].name == 'Graphene' and inputs['direction'] == 'Zigzag':
                filename = inputs['lattice']+'_ZGNR_'
            condition = 'Z='+str(thisUnit['Region'])+\
                        ',Type='+str(thisUnit['Type'])+\
                        ',S='+str(thisUnit['Shift'])+\
                        ',W='+str(thisUnit['W'])+\
                        ',L='+str(thisUnit['L'])+\
                        ',Vtop='+str(thisUnit['Vtop'])+\
                        ',Vbot='+str(thisUnit['Vbot'])+\
                        ',d='+str(thisUnit['delta'])
            mplot.savefig('figures/'+filename+condition+'.png')
        except:
            os.mkdir('figures')
            mplot.savefig('figures/'+filename+condition+'.png')
