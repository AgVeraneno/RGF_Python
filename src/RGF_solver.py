import sys, os, copy, time, warnings
sys.path.append('../lib/')
import numpy as np
import data_util
import lib_material, lib_excel, obj_unit_cell, IO_util, cal_band, cal_RGF
from multiprocessing import Pool

if __name__ == '__main__':
    '''
    This program simulates ballistic transpotation along x-axis.
    '''
    try:
        '''
        Work with command line.
        use "python RGF_solver.py <setup file> <job file>" command
        
        for excel input, setup file contains "__setup__" and "job" sheets.
        <job file> is not needed
        '''
        p_name, p_type = data_util.string_splitter(sys.argv[0],'.')
        input_file, input_type = data_util.string_splitter(sys.argv[1],'.')
        print('Program ',p_name, 'start @ ',time.asctime(time.localtime(time.time())))
        t_start = time.time()       # record import time
        if input_type == 'xlsx' or input_type == 'xls':
            inputs = IO_util.importFromExcel('../input/'+sys.argv[1])
        elif input_type == 'csv':
            setup, jobs = IO_util.importFromCSV('../input/'+sys.argv[1],
                                                '../input/'+sys.argv[2])
        else:
            raise ValueError('Not supported type input:',input_type)
    except:
        print('Program RGF_solver start @ ',time.asctime(time.localtime(time.time())))
        input_type = input('please provide input type:')
        t_start = time.time()       # record import time
        if input_type == 'xlsx' or input_type == 'xls':
            inputs = IO_util.importFromExcel('../input/RGF_input_file.xlsx')
        elif input_type == 'csv':
            setup, jobs = IO_util.importFromCSV('../input/RGF_setup.csv',
                                                '../input/RGF_job.csv')
        else:
            raise ValueError('Not supported type input:',input_type)
    print('Import time:', time.time() - t_start, '(sec)')
    '''
    Create unit cell
    '''
    t_start = time.time()       # record unit cell generation time
    unit_list = []              # unit cell object list
    for job in jobs:
        ###                                  ###
        # Add new type of simulation type here #
        ###                                  ###
        r_idx = int(job['region'])-1        # region index
        try:
            unitcell = unit_list[r_idx]
        except:
            if setup['brief'] == 'AGNR':
                new_unitcell = unit_cell.AGNR(setup['material'], job)
                
        new_unitcell.genHamiltonian()
        unit_list.append(new_unitcell)
        IO_util.saveAsExcel(inputs, idx, new_unitcell, save_type='matrix')
    t_unitGen = time.time() - t_start
    print('Generate unit cell time:',t_unitGen,'(sec)')
    '''
    Calculate band structure
    '''
    t_start = time.time()       # record band structure time
    #bs_parser = bs_GPU.BandStructure(inputs)
    if inputs['function']['isPlotBand']:
        for u_idx, unit in enumerate(unit_list):
            band_parser = cal_band.CPU(inputs, unit)
            bandgap_list = {'x':[],'y':[]}
            # construct each unit cell
            for idx in range(0,inputs['kx_mesh'],int(inputs['kx_mesh']/500)):
                if inputs['GPU']['enable']:
                    val, vec = bs_parser.calState_GPU(unit, idx)
                else:
                    val, vec = band_parser.calState(idx)
                ## record data ##
                
                bandgap_list['x'].append(unit.kx_norm)
                bandgap_list['y'].append([])
                for j in range(6):
                    bandgap_list['y'][-1].append(val[unit.m_MLG-4+j])
                    
            IO_util.saveAsFigure(inputs, u_idx, unit, bandgap_list, save_type='band')
    t_band = time.time() - t_start
    print('Calculate band structure:',t_band,'(sec)')
    '''
    Construct Green's matrix
    '''
    t_start = time.time()
    if inputs['GPU']['enable']:
        import cal_RGF_GPU
        RGF_util = cal_RGF_GPU.GPU(inputs, unit_list)
    else:
        RGF_util = cal_RGF.CPU(inputs, unit_list)
    
    kx_sweep = range(inputs['mesh'][0],inputs['mesh'][1])
    RGF_result = {'E':[],'T':[],'R':[]}
    if inputs['CPU']['p_enable'] and not inputs['GPU']['enable']:
        with Pool(processes=inputs['CPU']['p_num']) as mp:
            RGF_output = mp.map(RGF_util.calRGF_transmit,kx_sweep)
        RGF_result['E'] = np.array(RGF_output)[:,0]
        RGF_result['T'] = np.array(RGF_output)[:,1]
        RGF_result['R'] = np.array(RGF_output)[:,2]
    else:
        for kx_idx in range(inputs['mesh'][0],inputs['mesh'][1]):        # sweep kx meshing
            ## RGF
            RGF_output = RGF_util.calRGF_transmit(kx_idx)
            RGF_result['E'].append(RGF_output[0])
            RGF_result['T'].append(RGF_output[1])
            RGF_result['R'].append(RGF_output[2])
    ## plot transmission ##
    unit = unit_list[0]
    IO_util.saveAsFigure(inputs, -1, unit, RGF_result, save_type='TR')
    IO_util.saveAsExcel(inputs, -1, unit, RGF_result, save_type='TR')
    t_RGF = time.time() - t_start
    print('Calculate RGF:',t_RGF,'(sec)')