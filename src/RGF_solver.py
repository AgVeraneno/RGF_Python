import sys, os, copy, time, warnings
import numpy as np
from multiprocessing import Pool
import data_util, IO_util
import unit_cell, unit_cell_graphene, cal_band, cal_RGF

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
    t_load = time.time()
    ## build up folder if not exist
    input_dir = '../input/'
    output_dir = '../output/'
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    # resolve user inputs
    if len(sys.argv) == 1:
        setup_file = input_dir+'RGF_setup.csv'       # load default setup file
        isGPU = False
        workers = 1
    else:
        # input file
        if '-i' in sys.argv:
            setup_file = sys.argv[sys.argv.index('-i') +1]
        else:
            setup_file = input_dir+'RGF_setup.csv'       # load default setup file
        # GPU assisted RGF
        if '-gpu' in sys.argv:
            isGPU = True
        else:
            isGPU = False
        # Parallel CPU count
        if '-turbo' in sys.argv:
            workers = int(sys.argv[sys.argv.index('-turbo') +1])
        else:
            workers = 1
    # check inputs
    if not os.path.exists(setup_file):
        raise ValueError('Invalid input file: ',setup_file)
    # load setup file
    setup_dict, job_dict = IO_util.load_setup(setup_file)
    t_load = round(time.time() - t_load,3)
    print('Import time:', t_load, '(sec)')
    t_total += t_load
    ############################################################
    # Run simulation
    # 1. Create splits of a single job
    # 2. Generate unit cell
    # 3. Calculate band diagram
    # 4. Calculate RGF
    ############################################################
    for job_name, job in job_dict.items():
        ## make directory
        job_dir = output_dir+job_name
        if not os.path.exists(job_dir): os.mkdir(job_dir)
        '''
        Create splits
        '''
        split_table = []
        job_sweep = {}
        for region in job['region list']:
            job_sweep[region] = []
            for var_idx, var in enumerate(job[region]['sweep_var']):
                if len(var) > 0:    # split enabled
                    sweep_list = data_util.str2float1D(var, totem=';', dtype='str')
                    sweep_val = data_util.str2float1D(job[region]['sweep_val'][var_idx], totem=';', dtype='str')
                    for val_idx, val in enumerate(sweep_val):
                        swp_typ, swp_val = data_util.str2float1D(val,totem='&')
                        sweep_val[val_idx] = [swp_typ, swp_val]
                    sweep_dict = {}
                    for val_idx, vals in enumerate(sweep_val):
                        # split string to numbers
                        val = data_util.str2float1D(vals[1],totem=',')
                        val_type = vals[0]
                        vals = []
                        for v in val:
                            if isinstance(v, str):
                                # linspace type input
                                v = data_util.str2float1D(v,totem=':')
                                v = np.arange(v[0],v[2],v[1])
                                vals.extend(v)
                            else:
                                vals.append(v)
                        else:
                            sweep_dict[sweep_list[val_idx]] = {'type':val_type,
                                                               'value':vals}
                    else:
                        job_sweep[region].append(sweep_dict)
                else:
                    job_sweep[region].append({})
        else:
            # generate split table
            for s_key, split in job_sweep.items():
                for r_idx, sub_unit in enumerate(split):
                    for key, var in sub_unit.items():
                        if var['type'] == 'var':
                            for v in var['value']:
                                new_job = copy.deepcopy(job)
                                new_job[s_key][key][r_idx] = v
                                split_table.append(new_job)
                        elif var['type'] == 'sync':
                            for idx, old_job in enumerate(split_table):
                                old_job[s_key][key][r_idx] = var['value'][idx%len(var['value'])]
            else:
                if len(split_table) == 0:
                    split_table.append(job)
        '''
        Calculate splits
        '''
        split_summary = {}
        for s_idx, split in enumerate(split_table):
            split_summary[s_idx] = []
            t_split = 0
            t_unitcell = time.time()
            ## resolve calculation condition
            k0, kN = data_util.str2float1D(split['kx'], totem=',', dtype='int')
            CB_raw = data_util.str2float1D(split['CB'], totem=',', dtype='int')
            kx_list = range(k0,kN+1)
            CB_list = []
            for idx, CB in enumerate(CB_raw):
                if isinstance(CB, str):
                    a,b,c = data_util.str2float1D(CB, totem=':')
                    CB_list.extend(np.arange(a,c+b,b,dtype=int))
                else:
                    CB_list.append(CB)
            '''
            Generate unit cell
            '''
            unit_list = {}
            for r_idx, region in enumerate(split['region list']):
                if setup_dict['structure'] == 'AGNR':
                    unit_list[region] = unit_cell_graphene.AGNR(setup_dict, split[region])
                elif setup_dict['structure'] == 'ATNR':
                    unit_list[region] = unit_cell.ATNR(setup_dict, split[region])
                elif setup_dict['structure'] == 'ATNR10':
                    unit_list[region] = unit_cell.ATNR10(setup_dict, split[region])
                else:
                    raise ValueError('Non supported setup:',setup['structure'])
            ## print out Hamiltonian in debug mode
            if setup_dict['debug']:
                ## build debug folder
                folder = job_dir+'/debug/'
                if not os.path.exists(folder): os.mkdir(folder)
                for r_key, region in unit_list.items():
                    IO_util.saveAsCSV(folder+str(s_idx)+'_'+r_key+'_H.csv', region.H)
                    IO_util.saveAsCSV(folder+str(s_idx)+'_'+r_key+'_P+.csv', region.Pf)
                    IO_util.saveAsCSV(folder+str(s_idx)+'_'+r_key+'_P-.csv', region.Pb)
            t_unitcell = round(time.time() - t_unitcell,3)
            #print('Unit cell:', t_unitcell, '(sec)')
            t_split += t_unitcell
            '''
            Calculate band diagram
            '''
            t_band = time.time()
            if setup_dict['band']:
                for key, unit in unit_list.items():
                    ## initialize ##
                    band_parser = cal_band.CPU(setup_dict, unit)
                    sweep_mesh = range(0,int(setup_dict['kx_mesh']),1)
                    ## calculate band structure ##
                    #with Pool(processes=workers) as mp:
                    #    eig = mp.map(band_parser.calState,sweep_mesh)
                    eig = []
                    for i in sweep_mesh:
                        if i == 0:
                            eig.append(band_parser.calState(i))
                        else:
                            eig.append(band_parser.calState(i, ref_val=eig[i-1][1], ref_vec=eig[i-1][3]))
                    ## output eigenvalues
                    # build plot table header
                    plot_table = [['kx']]
                    for idx in range(np.size(eig[0][1])):
                        plot_table[0].append('E'+str(idx+1)+' (eV)')
                    # build data table
                    for i in eig:
                        plot_table.append([i[0]])
                        plot_table[-1].extend(list(np.real(i[1])))
                    ## output eigenvectors
                    # build state table header
                    state_table = {}
                    for kx in kx_list:
                        state_table[kx-1] = [['2y/a']]
                        for CB in CB_list:
                            for idx in range(unit.SU_size):
                                state_table[kx-1][0].append('CB='+str(CB)+'('+setup_dict['header'][idx%unit.SU_size]+')')
                            else:
                                state_table[kx-1][0].append("$")
                    # build sorted state table header
                    sorted_state_table = {}
                    for CB in CB_list:
                        sorted_state_table[CB] = [['kx']]
                        if setup_dict['lattice'] == 'MLG':
                            for idx in range(np.size(eig[0][1],0)):
                                sorted_state_table[CB][0].append('L1_'+str(idx))
                        elif setup_dict['lattice'] == 'BLG':
                            for idx in range(int(np.size(eig[0][1],0)/2)):
                                sorted_state_table[CB][0].append('L1_'+str(idx))
                            for idx in range(int(np.size(eig[0][1],0)/2)):
                                sorted_state_table[CB][0].append('L2_'+str(idx))
                        for i in eig:
                            sorted_state_table[CB].append([i[0]])
                            sorted_state_table[CB][-1].extend(list(abs(i[3][:,CB])))
                                

                    
                    for kx in kx_list:
                        num_of_item = int(len(eig[0][2][:,0])/unit.SU_size)
                        for idx in range(num_of_item):
                            state_table[kx-1].append([idx+1])
                            for CB in CB_list:
                                state_table[kx-1][-1].extend(list(np.abs(eig[kx-1][2][idx*unit.SU_size:(idx+1)*unit.SU_size,CB-1])))
                                state_table[kx-1][-1].append('')
                        else:
                            tmp_table = copy.deepcopy(state_table[kx-1])
                            if setup_dict['SU_type'] == 'separate':
                                n_sep = 1
                                n_ovl = int(num_of_item/2)+num_of_item%2+1
                                for i in range(0,num_of_item,2):
                                    state_table[kx-1][i+1] = tmp_table[n_sep]
                                    state_table[kx-1][i+1][0] = i+1
                                    try:
                                        state_table[kx-1][i+2] = tmp_table[n_ovl]
                                        state_table[kx-1][i+2][0] = i+2
                                    except:
                                        pass
                                    n_sep += 1
                                    n_ovl += 1
                            elif setup_dict['SU_type'] == 'overlap':
                                n_sep = 1
                                n_ovl = int(num_of_item/2)+1
                                for i in range(0,num_of_item,2):
                                    state_table[kx-1][i+1] = tmp_table[n_ovl]
                                    state_table[kx-1][i+1][0] = i+1
                                    try:
                                        state_table[kx-1][i+2] = tmp_table[n_sep]
                                        state_table[kx-1][i+2][0] = i+2
                                    except:
                                        pass
                                    n_sep += 1
                                    n_ovl += 1
                    ## output to file
                    folder = job_dir+'/band/'
                    if not os.path.exists(folder):
                        os.mkdir(folder)
                    IO_util.saveAsCSV(folder+str(s_idx)+'_'+key+'_band.csv', plot_table)
                    for kx in kx_list:
                        IO_util.saveAsCSV(folder+str(s_idx)+'_'+key+'_eigenstates@kx='+str(kx)+'.csv', state_table[kx-1])
                    for CB in CB_list:
                        IO_util.saveAsCSV(folder+str(s_idx)+'_'+key+'_eigenstates@CB='+str(CB)+'.csv', sorted_state_table[CB])
                    '''
                    try:
                        IO_util.saveAsFigure(setup_dict, folder+key, unit, plot_table, save_type='band')
                    except:
                        warnings.warn("error when ploting figures. Skip and continue.")
                    '''
                else:
                    t_band = round(time.time() - t_band,3)
                    #print('Band diagram:', t_band, '(sec)')
                    t_split += t_band
            '''
            Calculate RGF
            '''
            t_RGF = time.time()
            if setup_dict['RGF']:
                folder = job_dir+'/PTR/'
                if not os.path.exists(folder):
                    os.mkdir(folder)
                RGF_header = ['kx |1/a|','Energy (eV)','Transmission('+setup_dict['spin'][0]+')','Transmission('+setup_dict['spin'][1]+')','Transmission(Total)']
                RGF_util = cal_RGF.CPU(setup_dict, unit_list)
                CB_cache = {}
                for CB in CB_list:
                    RGF_util.CB = CB-1
                    RGF_util.C0 = []
                    RGF_util.CN = []
                    with Pool(processes=workers) as mp:
                        RGF_output = mp.map(RGF_util.calRGF_transmit,kx_list)
                    RGF_output = np.real(RGF_output)
                    ## sort kx position low to high
                    RGF_output_sort = RGF_util.sort_E(RGF_output)
                    CB_cache[CB] = copy.deepcopy(RGF_output_sort[:,2:6])
                    RGF_output_sort[:,2] = RGF_output_sort[:,2]/RGF_output_sort[:,-1]
                    RGF_output_sort[:,3] = RGF_output_sort[:,3]/RGF_output_sort[:,-1]
                    RGF_output_sort[:,4] = RGF_output_sort[:,4]/RGF_output_sort[:,-1]
                    ## add header
                    RGF_tmp = np.zeros((np.size(RGF_output_sort,0)+1,np.size(RGF_output_sort,1)-1), dtype=np.object)
                    RGF_tmp[0,:] = RGF_header
                    RGF_tmp[1:,:] = RGF_output_sort[:,:-1]
                    split_summary[s_idx].append(RGF_output_sort[:,:-1])
                    ## output to file ##
                    IO_util.saveAsCSV(folder+str(s_idx)+'_CB='+str(CB)+'_TR.csv', RGF_tmp)
                    '''
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
                    '''
                else:
                    t_RGF = time.time() - t_RGF
                    #print('Calculate RGF:',t_RGF,'(sec)')
                    t_split += t_RGF
            print('Split_'+str(s_idx)+':',t_split,'(sec)')
            t_total += t_split
        else:
            '''
            Summary table
            '''
            if setup_dict['RGF']:
                ## generate header
                RGF_header = ['Split', 'CB', 'kx (1/a)','Energy (eV)']
                RGF_header.append('Local transmission('+setup_dict['spin'][0]+')')
                RGF_header.append('Local transmission('+setup_dict['spin'][1]+')')
                RGF_header.append('Local transmission(Total)')
                RGF_header.append('Couple transmission('+setup_dict['spin'][0]+')')
                RGF_header.append('Couple transmission('+setup_dict['spin'][1]+')')
                RGF_header.append('Couple transmission(Total)')
                RGF_table = []
                for CB_idx, CB in enumerate(CB_list):
                    for kx_idx, kx in enumerate(kx_list):
                        for s_idx, split in enumerate(split_table):
                            RGF_table.append(['Split_'+str(s_idx)])
                            RGF_table[-1].append(str(CB))
                            RGF_table[-1].extend(split_summary[s_idx][CB_idx][kx_idx,:])
                            RGF_table[-1].extend(CB_cache[CB_list[(CB_idx+1)%2]][kx_idx,:-1]/CB_cache[CB_list[(CB_idx)%2]][kx_idx,-1])
                else:
                    RGF_table = np.block(RGF_table)
                    RGF_tmp = np.zeros((np.size(RGF_table,0)+1,np.size(RGF_table,1)), dtype=np.object)
                    RGF_tmp[0,:] = RGF_header
                    RGF_tmp[1:,:] = RGF_table
                    IO_util.saveAsCSV(folder+'Split_summary.csv', RGF_tmp)
            print('Program finished successfully @ ',time.asctime(time.localtime(time.time())))
            print('Total time: ', round(t_total,3), ' (sec)')
"""
            
    try:
        '''
        Work with command line.
        use "python RGF_solver.py <setup file> <job file>" command
        
        for excel input, setup file contains "__setup__" and "job" sheets.
        <job file> is not needed
        '''
        p_name, p_type = data_util.str2float1D(sys.argv[0],'.')
        input_file, input_type = data_util.str2float1D(sys.argv[1],'.')
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
"""