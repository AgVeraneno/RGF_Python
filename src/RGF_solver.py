import sys, os, copy, time, warnings
sys.path.append('../lib/')
import numpy as np
import data_util
import unit_cell
import IO_util, cal_band, cal_RGF
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
    job_list = {}
    unit_list = {}              # unit cell object list
    ## combine jobs ##
    for job in jobs:
        if job['region'] in job_list:
            job_list[job['']] = 
        else:
            job_list[job['region']] = job
    ## bulid unit cell ##
    for job in job_list:
        ###                                  ###
        # Add new type of simulation type here #
        ###                                  ###
        
        try:
            unitcell = unit_list[job['region']]
            unitcell.updateHamiltonian(setup, job)
        except:
            ## currently support AGNR only
            if setup['brief'] == 'AGNR':
                unitcell = unit_cell.AGNR(setup, job)
            else:
                raise ValueError('Non supported setup:',setup['brief'])
            unit_list[job['region']] = unitcell
    #{ for debug use
    for u_idx,unit in unit_list.items():
        folder = '../output/debug/'
        if not os.path.exists(folder):
            os.mkdir(folder)
        file_name = unit.info['region']+'_'+unit.info['lattice']+'_'+unit.info['brief']
        IO_util.saveAsCSV(folder+file_name+'_H.csv', unit.H)
        IO_util.saveAsCSV(folder+file_name+'_P+.csv', unit.P_plus)
        IO_util.saveAsCSV(folder+file_name+'_P-.csv', unit.P_minus)
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
        lead_unit = unit_list['lead']
        band_parser = cal_band.CPU(setup, lead_unit)
        sweep_mesh = range(0,int(setup['kx_mesh']),int(int(setup['kx_mesh'])/500))
        ## calculate band structure ##
        with Pool(processes=int(setup['parallel_CPU'])) as mp:
            eig = mp.map(band_parser.calState,sweep_mesh)
        plot_table = []
        for i in eig:
            plot_table.append([i[0]])
            plot_table[-1].extend(list(np.real(i[1])))
        ## output to file
        folder = '../output/band structure/'
        if not os.path.exists(folder):
            os.mkdir(folder)
        file_name = ''
        IO_util.saveAsCSV(folder+file_name+'_BS.csv', plot_table)
        #IO_util.saveAsFigure(setup, 0, lead_unit, plot_table, save_type='band')
    t_band = time.time() - t_start
    print('Calculate band structure:',t_band,'(sec)')
    '''
    Construct Green's matrix
    '''
    t_start = time.time()
    if setup['isRGF']:
        kx_sweep = range(int(setup['mesh_start']),int(setup['mesh_stop'])+1)
        RGF_result = {'E':[],'T':[],'R':[]}
        if setup['isGPU']:      # using GPU calculation
            ## not support GPU yet
            pass
        else:
            RGF_util = cal_RGF.CPU(setup, unit_list)
            with Pool(processes=int(setup['parallel_CPU'])) as mp:
                RGF_output = mp.map(RGF_util.calRGF_transmit,kx_sweep)
            RGF_output = np.real(RGF_output)
        ## output to file ##
        folder = '../output/'
        if not os.path.exists(folder):
            os.mkdir(folder)
        file_name = ''
        IO_util.saveAsCSV(folder+file_name+'_TR.csv', RGF_output)
    t_RGF = time.time() - t_start
    print('Calculate RGF:',t_RGF,'(sec)')
    print('Program stop @ ',time.asctime(time.localtime(time.time())))