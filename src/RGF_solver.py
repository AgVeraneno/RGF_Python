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
              'function':{'isPlotBand':False,
                          'isPlotZoom':False},
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
            elif row[0].value == 'mesh':
                inputs['mesh'][0] = int(row[1].value)
                inputs['mesh'][1] = int(row[2].value)
            elif row[0].value == 'Bias(V)':
                inputs['Vbias'][0] = float(row[1].value)
                inputs['Vbias'][1] = float(row[2].value)
            elif row[0].value == 'Plot band structure':
                inputs['function']['isPlotBand'] = bool(row[1].value)
                inputs['function']['isPlotZoom'] = bool(row[2].value)
            elif row[0].value == 'o':
                new_unit = {'Region': int(row[1].value),
                            'Type': int(row[2].value),
                            'Shift': int(row[3].value),
                            'W': int(row[4].value),
                            'L': int(row[5].value),
                            'Vtop': float(row[6].value),
                            'Vbot': float(row[7].value),
                            'delta': float(row[8].value),
                            'Barrier':{'top width':int(row[9].value),
                                       'top gap':float(row[10].value),
                                       'bot width':int(row[11].value),
                                       'bot gap':float(row[12].value)}}
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
        warnings.warn('No valid PyQt4 installed. This may cause error!')
    '''
    This program simulates ballistic transpotation along x-axis.
    '''
    t_start = time.time()       # record import time
    inputs = importSetting('RGF_input_file.xlsx')
    t_import = time.time() - t_start
    print('Import time:', t_import, '(sec)')
    '''
    Generate RGF unit cell
    '''
    t_start = time.time()       # record unit cell generation time
    unit_list = []
    for idx in range(len(inputs['Unit cell'])):
        new_unitcell = obj_unit_cell.UnitCell(inputs, idx)
        new_unitcell.genHamiltonian()
        unit_list.append(new_unitcell)
        new_unitcell.saveAsXLS()
    t_unitGen = time.time() - t_start
    print('Generate unit cell time:',t_unitGen,'(sec)')
    '''
    calculate band structure and incident state
    use CPU only!!
    '''
    t_start = time.time()       # record band structure time
    band_structure = bs.BandStructure(inputs)
    if inputs['function']['isPlotBand']:
        for unit_idx, unit in enumerate(unit_list):
            bandgap_list = {'x':[],'y':[]}
            for idx in range(inputs['mesh'][1]-1):
                val, vec = band_structure.calState(unit, idx)
                bandgap_list['y'].append(val)
                bandgap_list['x'].append(unit.kx_norm)
                ## record incident state
                if unit_idx == 0 and idx == 0:
                    ## find 1st conduction band ##
                    for v_idx, v in enumerate(val):
                        if round(v-unit.info['delta'],5) == 0:
                            band_idx = v_idx
                            break
                        else:
                            pass
                    En0 = copy.deepcopy(val)
                    i_state = copy.deepcopy(vec[:,v_idx])
            band_structure.plotBand(bandgap_list, unit_idx)
    else:
        ## record incident state
        unit = unit_list[0]
        val, vec = band_structure.calState(unit, idx)
        ## find 1st conduction band ##
        for v, v_idx in enumerate(val):
            if round(v-unit.info['delta'],5) == 0:
                band_idx = v_idx
                break
        En0 = copy.deepcopy(val[v_idx])
        i_state = copy.deepcopy(vec[:,v_idx])
    t_band = time.time() - t_start
    print('Calculate band structure:',t_band,'(sec)')
    t_start = time.time()
    '''
    Construct Green's matrix
    '''
    Vbias = (inputs['Vbias'][0] - inputs['Vbias'][1])*1.6e-19
    ## Calculate RGF ##
    RGF_util = gf.GreenFunction(inputs, unit_list)
    for E_idx, E in enumerate(En0):
        if inputs['GPU']['enable']:
            if E >= 1e-25 and E <= Vbias:
                RGF_util.calRGF_GPU(E)
                ## Calculate transmission/reflection current
                RGF_util.calState_GPU(i_state[:,E_idx], o_state[:,E_idx])
                Jt, Jr = RGF_util.calTR_GPU()
                print(E/1.6e-19, Jt, Jr)
            else:
                Jt = Jr = 0
        else:
            if E >= 1e-25 and E <= Vbias:
                RGF_util.calRGF(E)
                ## Calculate transmission/reflection current
                RGF_util.calState(i_state[:,E_idx], o_state[:,E_idx])
                Jt, Jr = RGF_util.calTR()
                print(E/1.6e-19, Jt, Jr)
            else:
                Jt = Jr = 0
        