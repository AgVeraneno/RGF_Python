import sys, os, copy, time, warnings
import numpy as np
from multiprocessing import Pool
import data_util, IO_util
import unit_cell, cal_band, cal_RGF

if __name__ == '__main__':
    print('Start RGF solver @ ',time.asctime(time.localtime(time.time())))
    t_total = 0
    '''
    This program simulates ballistic transportation along x-axis.
    '''
    ############################################################
    # Environment setup
    # 1. build up "output" folder.
    # 2. get user's inputs. Input type using sys.argv.
    ############################################################
    ## build up folder if not exist
    input_dir = '../input/'
    output_dir = '../output/'
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    # resolve user inputs
    if len(sys.argv) == 1:
        job_file = input_dir+'RGF_job.csv'       # load default setup file
    else:
        # input file
        if '-i' in sys.argv:
            job_file = sys.argv[sys.argv.index('-i') +1]
        else:
            job_file = input_dir+'RGF_job.csv'       # load default setup file
        # GPU assisted RGF
        if '-gpu' in sys.argv:
            isGPU = True
        else:
            isGPU = False
        # Parallel CPU count
        if '-turbo' in sys.argv:
            workers = int(sys.argv[sys.argv.index('-i') +1])
        else:
            workers = 1
    # check inputs
    if not os.path.exists(job_file):
        raise ValueError('Invalid input file: ',job_file)
    # load setup file
    jobs = IO_util.importFromCSV()
    
    
    
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
        #input_type = input('please provide input type:')
        input_type = 'csv'
        t_start = time.time()       # record import time
        if input_type == 'xlsx' or input_type == 'xls':
            inputs = IO_util.importFromExcel('../input/RGF_input_file.xlsx')
        elif input_type == 'csv':
            setup, jobs = IO_util.importFromCSV('../input/RGF_setup.csv',
                                                '../input/RGF_job.csv')
        else:
            raise ValueError('Not supported type input:',input_type)
    print('Import time:', time.time() - t_start, '(sec)')
    t_total += time.time() - t_start
    '''
    Create unit cell
    '''
    t_start = time.time()       # record unit cell generation time
    split_list = {}             # split for jobs
    job_list = {}               # unit cell definition
    unit_list = {}              # unit cell object list
    ## combine jobs ##
    for job in jobs:
        split_key = job['name']
        key = job['region']
        if not split_key in split_list:
            new_job = {}
            new_job['region'] = str(job['region'])
            new_job['type'] = [str(job['cell_type'])]
            new_job['shift'] = [int(job['shift'])]
            new_job['width'] = [int(job['width'])]
            new_job['Vtop'] = [float(job['Vtop'])]
            new_job['Vbot'] = [float(job['Vbot'])]
            new_job['gap'] = [float(job['gap'])]
            new_job['length'] = int(float(job['length']))
            split_list[split_key] = {key:new_job}
        elif not key in split_list[split_key]:
            new_job = {}
            new_job['region'] = str(job['region'])
            new_job['type'] = [str(job['cell_type'])]
            new_job['shift'] = [int(job['shift'])]
            new_job['width'] = [int(job['width'])]
            new_job['Vtop'] = [float(job['Vtop'])]
            new_job['Vbot'] = [float(job['Vbot'])]
            new_job['gap'] = [float(job['gap'])]
            new_job['length'] = int(float(job['length']))
            split_list[split_key][key] = new_job
        elif key in split_list[split_key]:
            split_list[split_key][key]['type'].append(str(job['cell_type']))
            split_list[split_key][key]['shift'].append(int(job['shift']))
            split_list[split_key][key]['width'].append(int(job['width']))
            split_list[split_key][key]['Vtop'].append(float(job['Vtop']))
            split_list[split_key][key]['Vbot'].append(float(job['Vbot']))
            split_list[split_key][key]['gap'].append(float(job['gap']))
    ###                                  ###
    # Add new type of simulation type here #
    ###                                  ###
    
    for splitID, split in split_list.items():
        folder = output_dir+splitID+'/'
        if not os.path.exists(folder):
            os.mkdir(folder)
        for jobid, job in split.items():
            if setup['brief'] == 'AGNR':
                unit_list[job['region']] = unit_cell.AGNR_new(setup, job)
            elif setup['brief'] == 'AMNR':
                unit_list[job['region']] = unit_cell.AMNR_new(setup, job)
            else:
                raise ValueError('Non supported setup:',setup['brief'])
        ## print out matrix. debug use ##
        if setup['isDebug']:
            for u_idx,unit in unit_list.items():
                folder = output_dir+splitID+'/debug/'
                if not os.path.exists(folder):
                    os.mkdir(folder)
                file_name = unit.filename
                IO_util.saveAsCSV(folder+file_name+'_H.csv', unit.H)
                IO_util.saveAsCSV(folder+file_name+'_P+.csv', unit.Pf)
                IO_util.saveAsCSV(folder+file_name+'_P-.csv', unit.Pb)
        t_unitGen = time.time() - t_start
        print('\nGenerate unit cell time:',t_unitGen,'(sec)')
        t_total += time.time() - t_start
        '''
        Calculate band structure
        '''
        t_start = time.time()       # record band structure time
        #bs_parser = bs_GPU.BandStructure(inputs)
        if setup['isPlot_band']:
            for key, unit in unit_list.items():
                ## initialize ##
                band_parser = cal_band.CPU(setup, unit)
                sweep_mesh = range(0,int(setup['kx_mesh']),int(int(setup['kx_mesh'])/500))
                ## calculate band structure ##
                with Pool(processes=int(setup['parallel_CPU'])) as mp:
                    eig = mp.map(band_parser.calState,sweep_mesh)
                plot_table = []
                for i in eig:
                    plot_table.append([i[0]])
                    plot_table[-1].extend(list(np.real(i[1])))
                ## output to file
                folder = output_dir+splitID+'/band structure/'
                if not os.path.exists(folder):
                    os.mkdir(folder)
                IO_util.saveAsCSV(folder+key+'_BS.csv', plot_table)
                try:
                    IO_util.saveAsFigure(setup, folder+key, unit, plot_table, save_type='band')
                except:
                    warnings.warn("error when ploting figures. Skip and continue.")
        t_band = time.time() - t_start
        print('Calculate band structure:',t_band,'(sec)')
        t_total += time.time() - t_start
        '''
        Construct Green's matrix
        '''
        t_start = time.time()
        if setup['isRGF']:
            kx_sweep = range(int(setup['mesh_start']),int(setup['mesh_stop'])+1)
            CB_idx = np.arange(int(setup['CB_idx_start'])-1,int(setup['CB_idx_stop']),1)
            RGF_header = ['kx |K|','Energy (eV)','Transmission(CN1)','Transmission(CN2)']
            if setup['isGPU']:      # using GPU calculation
                ## not support GPU yet
                pass
            else:
                RGF_util = cal_RGF.CPU(setup, unit_list)
                for CB in CB_idx:
                    RGF_util.CB = CB
                    with Pool(processes=int(setup['parallel_CPU'])) as mp:
                        RGF_output = mp.map(RGF_util.calRGF_transmit,kx_sweep)
                    RGF_output = np.real(RGF_output)
                    ## sort kx position low to high
                    RGF_output = RGF_util.sort_E(RGF_output)
                    ## add header
                    RGF_tmp = np.zeros((np.size(RGF_output,0)+1,np.size(RGF_output,1)), dtype=np.object)
                    RGF_tmp[0,:] = RGF_header
                    RGF_tmp[1:,:] = RGF_output
                    ## output to file ##
                    folder = output_dir+splitID+'/'
                    file_name = "CB="+str(CB+1)
                    IO_util.saveAsCSV(folder+file_name+'_TR.csv', RGF_tmp)
                    if setup['isReflect']:
                        RGF_util.reflect = True
                        with Pool(processes=int(setup['parallel_CPU'])) as mp:
                            RGF_output = mp.map(RGF_util.calRGF_transmit,kx_sweep)
                        RGF_output = np.real(RGF_output)
                        ## sort kx position low to high
                        RGF_output = RGF_util.sort_E(RGF_output)
                        ## output to file ##
                        IO_util.saveAsCSV(folder+file_name+'_TR_reverse.csv', RGF_output)
                        RGF_util.reflect = False
        t_RGF = time.time() - t_start
        print('Calculate RGF:',t_RGF,'(sec)')
        t_total += time.time() - t_start
    print('Program finished successfully @ ',time.asctime(time.localtime(time.time())))
    print('Total time: ', round(t_total,3), ' (sec)')