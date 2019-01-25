import sys, os, copy, time, warnings
sys.path.append('../lib/')
import numpy as np
import data_util
import unit_cell
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
                unitcell = unit_cell.AGNR(setup, job)
            else:
                raise ValueError('Non supported setup:',setup['brief'])
            unit_list.append(unitcell)
        #{ for debug use
        IO_util.saveAsCSV(setup, job, unitcell.H, 'H', '../output/')
        IO_util.saveAsCSV(setup, job, unitcell.P_plus, 'P+', '../output/')
        IO_util.saveAsCSV(setup, job, unitcell.P_minus, 'P-', '../output/')
        #}
    t_unitGen = time.time() - t_start
    print('Generate unit cell time:',t_unitGen,'(sec)')
    '''
    Calculate band structure
    '''
    t_start = time.time()       # record band structure time
    #bs_parser = bs_GPU.BandStructure(inputs)
    if setup['isPlot_band']:
        ## initialize ##
        lead_unit = unit_list[0]
        band_parser = cal_band.CPU(setup, lead_unit)
        sweep_mesh = range(0,int(setup['kx_mesh']),int(int(setup['kx_mesh'])/500))
        ## calculate band structure ##
        with Pool(processes=int(setup['parallel_CPU'])) as mp:
            eig = mp.map(band_parser.calState,sweep_mesh)
        plot_table = []
        for i in eig:
            plot_table.append([i[0]])
            plot_table[-1].extend(list(np.real(i[1])))
        IO_util.saveAsCSV(setup, jobs[0], plot_table, 'band', '../output/')
        IO_util.saveAsFigure(setup, 0, lead_unit, plot_table, save_type='band')
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