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
            #global cp
            #from skcuda import linalg as cp
            global cp
            import cupy as cp
            pass
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
    Create unit cell
    '''
    t_start = time.time()       # record unit cell generation time
    unit_list = []              # unit cell object list
    for idx in range(len(inputs['Unit cell'])):
        new_unitcell = obj_unit_cell.UnitCell(inputs, idx)
        new_unitcell.genHamiltonian()
        unit_list.append(new_unitcell)
        new_unitcell.saveAsXLS()
    t_unitGen = time.time() - t_start
    print('Generate unit cell time:',t_unitGen,'(sec)')
    '''
    Calculate band structure
    '''
    t_start = time.time()       # record band structure time
    band_parser = bs.BandStructure(inputs)
    for unit_idx, unit in enumerate(unit_list):
        bandgap_list = {'x':[],'y':[]}
        # construct each unit cell
        for idx in range(inputs['mesh'][1]):
            val, vec = band_parser.calState(unit, idx)
            bandgap_list['y'].append(val)
            bandgap_list['x'].append(unit.kx_norm)
            ## record incident state
            if unit_idx == 0 and idx == 0:
                ## find 1st conduction band ##
                CB_idx = band_parser.getCBidx(unit.info['delta'], val)
                En0 = copy.deepcopy(val[CB_idx])
                i_state = copy.deepcopy(vec[:,CB_idx])
        if inputs['function']['isPlotBand']:
            band_parser.plotBand(bandgap_list, unit_idx)
        unit.eig_state = copy.deepcopy(bandgap_list)
    t_band = time.time() - t_start
    print('Calculate band structure:',t_band,'(sec)')
    '''
    Construct Green's matrix
    '''
    t_start = time.time()
    RGF_util = gf.GreenFunction(inputs, unit_list)
    for kx_idx in range(inputs['mesh'][1]):        # sweep kx meshing
        if inputs['GPU']['enable']:
            ## RGF
            RGF_util.calRGF_GPU(En)
            ## Jt/JR
            Jt, Jr = RGF_util.calTR_GPU(i_state)
        else:
            ## RGF
            RGF_util.calRGF(kx_idx)
            ## Jt/JR
            Jt, Jr = RGF_util.calTR(i_state)
    ## plot transmission ##
    RGF_util.plotTR()
    RGF_util.saveAsXLS()
    t_RGF = time.time() - t_start
    print('Calculate RGF:',t_RGF,'(sec)')