import sys, os, copy, time, warnings
sys.path.append('../lib/')
import numpy as np
import lib_material, lib_excel, obj_unit_cell
import band_structure as bs
import Green_function as gf

def importSetting(filename=None):
    inputs = {'material': None,
              'lattice': None,
              'direction': None,
              'mesh': [0, 0],
              'Vbias': [0.0, 0.0],
              'Unit cell': [],
              'function':{'band structure':False},
              'CPU Max matrix':0,
              'GPU':{'enable': False,
                     'Max matrix':0}}
    with lib_excel.excel(filename) as excel_parser:
        for row in excel_parser.readSheet('__setup__'):
            if row[0].value == 'Using GPU':
                inputs['GPU']['enable'] = bool(row[1].value)
                inputs['GPU']['Max matrix'] = int(row[2].value)
            elif row[0].value == 'CPU max matrix':
                inputs['CPU Max matrix'] = int(row[1].value)
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
            elif row[0].value == 'Plot band structure':
                inputs['function']['band structure'] = bool(row[1].value)
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

if __name__ == '__main__':
    try:
        from PyQt4 import QtGui
        app = QtGui.QApplication(sys.argv)
    except:
        warnings.warn('No PyQt4 installed. This may cause error!')
    '''
    This program simulates ballistic transpotation along x-axis.
    '''
    inputs = importSetting('RGF_input_file.xlsx')
    '''
    Generate RGF unit cell
    '''
    unit_list = []
    for idx in range(len(inputs['Unit cell'])):
        new_unitcell = obj_unit_cell.UnitCell(inputs)
        new_unitcell.genHamiltonian(inputs['Unit cell'][idx])
        unit_list.append(new_unitcell)
    '''
    calculate band structure and incident state
    '''
    band_structure = bs.BandStructure(inputs)
    if inputs['function']['band structure']:
        for unit_idx, unit in enumerate(unit_list):
            bandgap_list = {'x':[],'y':[]}
            for idx in range(int(inputs['Unit cell'][unit_idx]['L']-1)):
                unit.setKx(idx)
                if inputs['GPU']['enable']:
                    val, vec = band_structure.calState_GPU(unit)
                    bandgap_list['y'].append(cp.sort(val))
                else:
                    val, vec = band_structure.calState(unit)
                    bandgap_list['y'].append(np.sort(val))
                bandgap_list['x'].append(unit.kx_norm)
                ## record incident state
                if unit_idx == 0 and idx == 0:
                    En0 = copy.deepcopy(val)
                    i_state = copy.deepcopy(vec)
            band_structure.plotBand(bandgap_list, unit_idx)
        ## record output state
        unit = unit_list[-1]
        unit.setKx(inputs['mesh'][1]-1)
        if inputs['GPU']['enable']:
            Enn, o_state = band_structure.calState_GPU(unit)
        else:
            Enn, o_state = band_structure.calState(unit)
    else:
        ## record incident state
        unit = unit_list[0]
        unit.setKx(0)
        if inputs['GPU']['enable']:
            En0, i_state = band_structure.calState_GPU(unit)
        else:
            En0, i_state = band_structure.calState(unit)
        ## record output state
        unit = unit_list[-1]
        unit.setKx(inputs['mesh'][1]-1)
        if inputs['GPU']['enable']:
            Enn, o_state = band_structure.calState_GPU(unit)
        else:
            Enn, o_state = band_structure.calState(unit)
    '''
    Construct Green's matrix
    '''
    ## Calculate RGF ##
    RGF_util = gf.GreenFunction(inputs, unit_list)
    for E_idx, E in enumerate(En0):
        if inputs['GPU']['enable']:
            if abs(E/1.6e-19) >= 1e-5:
                RGF_util.calRGF_GPU(E)
                ## Calculate transmission/reflection current
                RGF_util.calState_GPU(i_state[:,E_idx], o_state[:,E_idx])
                Jt, Jr = RGF_util.calTR_GPU()
        else:
            if round(E, 25) != 0:
                RGF_util.calRGF(E)
                ## Calculate transmission/reflection current
                RGF_util.calState(i_state[:,E_idx], o_state[:,E_idx])
                Jt, Jr = RGF_util.calTR()
        print(E/1.6e-19, Jt, Jr)