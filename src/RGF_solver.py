import sys, os, copy, time, warnings, logging
import numpy as np
from multiprocessing import Pool
import data_util, IO_util
import unit_cell, unit_cell_graphene, unit_cell_TMDc
import cal_band, cal_RGF

class RGF_solver():
    def __init__(self):
        ## start time counter
        self.t_total = 0
        ## default input/output folders
        self.input_dir = '../input/'
        self.output_dir = '../output/'
        if not os.path.exists(self.output_dir): os.mkdir(self.output_dir)
        # auto fill in with each split
        self.job_dir = ''
        self.kx_list = []
        self.CB_list = []
        ## resolve system inputs
        if len(sys.argv) >= 1:  # using command input
            # input file
            if '-i' in sys.argv: self.setup_file = sys.argv[sys.argv.index('-i') +1]
            else: self.setup_file = self.input_dir+'template.xlsx'       # load default setup file
            # GPU assisted RGF
            if '-gpu' in sys.argv: self.isGPU = True
            else: self.isGPU = False
            # Parallel CPU count
            if '-turbo' in sys.argv: self.workers = int(sys.argv[sys.argv.index('-turbo') +1])
            else: self.workers = 12
        ## check input file
        if not os.path.exists(self.setup_file):
            logger.error('Invalid input file: %s',self.setup_file)
            raise ValueError('Invalid input file: %s',self.setup_file)
    def load_inputs(self):
        t_load = time.time()
        if '.csv' in self.setup_file:
            setup_dict, job_dict, sweep_dict = IO_util.load_setup(self.setup_file)
        elif '.xlsx' in self.setup_file:
            setup_dict, job_dict, sweep_dict = IO_util.importFromExcel(self.setup_file)
            setup_dict['structure'] = setup_dict['Direction'][0]+setup_dict['Material name'][0]+'NR'
        ## create global parameters
        self.mesh = range(0,int(setup_dict['mesh']),1)
        logger.info('Import time: '+str(round(time.time() - t_load,3))+' (sec).')
        return setup_dict, job_dict, sweep_dict
    def create_splits(self, job):
        split_table = []
        job_sweep = {}
        for region in job['region list']:
            job_sweep[region] = []
            for var_idx, var in enumerate(job[region]['sweep_var']):
                if var != "":    # split is not empty
                    sweep_list = data_util.str2float1D(var, totem=';', dtype='str')
                    sweep_val = data_util.str2float1D(job[region]['sweep_val'][var_idx], totem=';', dtype='str')
                    for val_idx, val in enumerate(sweep_val):
                        swp_typ, swp_val = data_util.str2float1D(val,totem='&')
                        sweep_val[val_idx] = [swp_typ, swp_val]
                    sweep_dict = {}
                    for val_idx, vals in enumerate(sweep_val):
                        # split string to numbers
                        val_type = vals[0]
                        if val_type == 'fix': val = [float(vals[1])]
                        else: val = data_util.str2float1D(vals[1],totem=',')
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
                        elif var['type'] == 'fix':
                            for idx, old_job in enumerate(split_table):
                                old_job[s_key][key][r_idx] = var['value'][0]
            else:
                if len(split_table) == 0: split_table.append(job)   # no splits condition
                return split_table
    def create_splits_from_dict(self, sweep_dict):
        job_sweep = {}
        '''
        Resolve dictionary
        '''
        for key, val in sweep_dict.items():
            job_sweep[key] = []
            if key == 'POR':
                continue
            else:
                for sweep in val['Sweep_list']:
                    ## identify region
                    r, r_idx = data_util.str2float1D(sweep['Region'], totem='>', dtype='int')
                    ## identify variable
                    v, v_idx = data_util.str2float1D(sweep['var'], totem='>', dtype='str')
                    ## identify value
                    swp_val = data_util.str2float1D(sweep['val'], totem=',')
                    val_list = []
                    for val in swp_val:
                        if isinstance(val, str):
                            n0, dn, nn = data_util.str2float1D(val, totem=':')
                            for data in np.arange(n0,nn+dn,dn): val_list.append(data)
                        else:
                            val_list.append(float(val))
                    new_split = {'Region': r,
                                 'Layer': r_idx,
                                 'type': v,
                                 'sweep var': v_idx,
                                 'sweep val': val_list}
                    job_sweep[key].append(new_split)
        '''
        Generate split table
        '''
        job_sweep['split_table'] = {}
        for key, val in job_sweep.items():
            if key == 'split_table': continue
            split_table = []
            var_counter = 0
            for split in val:
                split_table.append([])
                if split['type'] == 'var':
                    ## variable type sweep
                    var_counter += 1
                    for v in split['sweep val']:
                        for v2 in range(var_counter):
                            split_table[-1].append(v)
                elif split['type'] == 'sync':
                    ## sync type sweep
                    for v in split['sweep val']:
                        for v2 in range(var_counter):
                            split_table[-1].append(v)
                elif split['type'] == 'fix':
                    for v2 in range(len(split_table[-2])):
                        split_table[-1].append(split['sweep val'][0])
                else:
                    logger.error('Invalid sweep type: %s', split['type'])
            else:
                job_sweep['split_table'][key] = split_table
        else:
            return job_sweep
    def resolve_mesh(self, mesh_list):
        m_list = data_util.str2float1D(mesh_list, totem=';', dtype='int')
        new_list = []
        for m_idx, m in enumerate(m_list):
            if isinstance(m, str):
                m0, mN = data_util.str2float1D(m, totem=':', dtype='int')
                for i in range(m0, mN+1):
                    new_list.append(i)
            else:
                new_list.append(m)
        else:
            return new_list
    def gen_unitCell(self, setup_dict, job):
        t_unitcell = time.time()
        unit_list = {}
        for r_name, region in job.items():
            if setup_dict['structure']+setup_dict['Option'] == 'AGNR':
                unit_list[r_name] = unit_cell_graphene.AGNR(setup_dict, region)
            elif setup_dict['structure']+setup_dict['Option'] == 'ZGNR':
                unit_list[r_name] = unit_cell_graphene.ZGNR(setup_dict, region)
            elif setup_dict['structure']+setup_dict['Option'] == 'ATNR':
                unit_list[r_name] = unit_cell_TMDc.ATNR6(setup_dict, region)
            elif setup_dict['structure']+setup_dict['Option'] == 'ATNR10':
                unit_list[r_name] = unit_cell_TMDc.ATNR10(setup_dict, region)
            elif setup_dict['Option'] == 'Square':
                unit_list[r_name] = unit_cell_graphene.Square(setup_dict, region)
            else:
                logger.critical('Non supported structure:'+setup_dict['structure'])
                raise ValueError('Non supported structure:',setup_dict['structure'])
        else:
            job_name = region['Job']
        ## print out Hamiltonian in debug mode
        if setup_dict['Debug']:
            ## build debug folder
            folder = self.job_dir+'/debug/'
            if not os.path.exists(folder): os.mkdir(folder)
            for r_key, region in unit_list.items():
                IO_util.saveAsCSV(folder+job_name+'_'+r_key+'_H.csv', region.H)
                IO_util.saveAsCSV(folder+job_name+'_'+r_key+'_P+.csv', region.Pf)
                IO_util.saveAsCSV(folder+job_name+'_'+r_key+'_P-.csv', region.Pb)
                IO_util.saveAsCSV(folder+job_name+'_'+r_key+'_uH.csv', region.uH)
                IO_util.saveAsCSV(folder+job_name+'_'+r_key+'_uP+.csv', region.uPf)
                IO_util.saveAsCSV(folder+job_name+'_'+r_key+'_uP-.csv', region.uPb)
                IO_util.saveAsCSV(folder+job_name+'_'+r_key+'_Yop.csv', region.__Yop__)
                IO_util.saveAsCSV(folder+job_name+'_'+r_key+'_Xop.csv', region.__Xop__)
                IO_util.saveAsCSV(folder+job_name+'_'+r_key+'_V.csv', region.V)
        t_unitcell = round(time.time() - t_unitcell,3)
        logger.info('Generate unit cell:'+str(t_unitcell)+'(sec)')
        self.t_total += t_unitcell
        return unit_list
    def cal_bandStructure(self, setup_dict, unit_list):
        '''
        Initialize
        '''
        t_band = time.time()
        ref_vec = []
        ref_val = []
        ref_weight = []
        '''
        Job start
        '''
        for key, unit in unit_list.items():
            ## initialize ##
            band_parser = cal_band.CPU(setup_dict, unit)
            ## calculate band structure
            with Pool(processes=self.workers) as mp: eig = mp.map(band_parser.calState,self.mesh)
            ## sort band and eigenstate
            # 1. Sort header
            for e_idx, e in enumerate(eig):
                if e_idx > 1:
                    srt_val, srt_vec = band_parser.__sort__(e[1],e[2],e[3],'weight', ref_wgt)
                    eig[e_idx] = (e[0], np.array(srt_val), np.array(srt_vec))
                    ref_wgt = copy.deepcopy(e[3])
                else: ref_wgt = copy.deepcopy(e[3])
            ## generate result table eigenvalues
            band_table = [['kx*a']]
            weight_table = [['Band','kx*a']]
            uTB = [['Band','kx*a','muTB(Re)','muTB(Im)','muTB(abs)']]
            I_loop = [['Band','kx*a','uB_tot']]
            for i in range(len(eig[0][1])):
                band_table[0].append('Band'+str(i+1)+' (eV)')
                weight_table[0].append('Site'+str(i+1))
            for i in range(int(len(eig[0][1])/2)):
                I_loop[0].append('Hex'+str(i+1))
            ## sort band and eigenstate
            for e_idx, e in enumerate(eig):
                kx = e[0]*band_parser.a
                val = e[1]/1.6e-19
                vec = e[2]
                # weight = e[3]
                '''
                ## sort eigenstate
                sorted_val, sorted_vec, sorted_wgt = band_parser.sort_eigenstate(val,vec,weight,ref_val,ref_vec,ref_weight)
                if e_idx >= 1 and setup_dict['Direction'] == 'AC':
                    ref_vec = sorted_vec
                    ref_val = sorted_val
                    ref_weight = sorted_wgt
                '''
                ## append data to table
                band_table.append([kx])
                band_table[-1].extend(np.real(val))
                for E_idx in unit.region['E_idx']:
                    if e_idx in unit.region['S_idx']:
                        ## eigenstate weight table
                        weight_table.append([E_idx])
                        weight_table[-1].append(kx)
                        weight_table[-1].extend(abs(vec[:,E_idx])**2)
                        #weight_table[-1].extend(np.real(sorted_vec[:,E_idx]))
                        if setup_dict['POR Magnetic moment']:
                            ## magnetic moment
                            uTB.append([])
                            uTB[-1].append(E_idx)
                            uTB[-1].append(kx)
                            uTB_val = band_parser.calMagneticMoment(kx/band_parser.ax, sorted_vec[:,E_idx], sorted_vec[:,E_idx])
                            uTB[-1].append(np.real(uTB_val))
                            uTB[-1].append(np.imag(uTB_val))
                            uTB[-1].append(np.abs(uTB_val))
                            ## moment current
                            I_loop.append([])
                            I_loop[-1].append(E_idx)
                            I_loop[-1].append(kx)
                            I_list = band_parser.calMagneticMomentCurrent(sorted_vec[:,E_idx])
                            I_loop[-1].append(sum(I_list))
                            I_loop[-1].extend(I_list)
            else:
                ## print out report
                folder = self.output_dir+self.job_dir+'/band/'
                if not os.path.exists(folder): os.mkdir(folder)
                IO_util.saveAsCSV(folder+self.job_name+'_'+key+'_band.csv', band_table)
                IO_util.saveAsCSV(folder+self.job_name+'_'+key+'_weight.csv', weight_table)
                IO_util.saveAsCSV(folder+self.job_name+'_'+key+'_uTB.csv', uTB)
                IO_util.saveAsCSV(folder+self.job_name+'_'+key+'_Iloop.csv', I_loop)
                break
        else:
            '''
            Job finish
            '''
            t_band = round(time.time() - t_band,3)
            logger.info('  Calculate band structure:'+str(t_band)+'(sec)')
            self.t_total += t_band
    def cal_RGF_transmission(self, setup_dict, unit_list, E_list, S_list, split_summary, s_idx):
        t_RGF = time.time()
        folder = self.job_dir+'/PTR/'
        if not os.path.exists(folder): os.mkdir(folder)
        RGF_header = ['kx |1/a|','Energy (eV)','Transmission(K)','Transmission(K)','Transmission(Total)']
        RGF_util = cal_RGF.CPU(setup_dict, unit_list)
        CB_cache = {}
        for CB in E_list:
            RGF_util.CB = CB-1
            RGF_util.C0 = []
            RGF_util.CN = []
            if setup_dict['GPU enable']:
                RGF_output = []
                for kx_idx in S_list:
                    RGF_output.append(RGF_util.calRGF_transmit(kx_idx))
                else:
                    RGF_output = np.array(RGF_output)
            else:
                with Pool(processes=self.workers) as mp:
                    RGF_output = mp.map(RGF_util.calRGF_transmit,S_list)
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
        t_RGF = round(time.time() - t_RGF,3)
        logger.info('  Calculate RGF (DC):'+str(t_RGF)+'(sec)')
        self.t_total += t_RGF
        return CB_cache, split_summary
    def cal_TDNEGF(self, setup_dict, unit_list, split_summary):
        t = time.time()
        if setup_dict['TD']:
            pass
    def gen_summary(self, setup_dict, CB_cache, split_summary):
        if setup_dict['RGF']:
            folder = self.job_dir+'/PTR/'
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
        
if __name__ == '__main__':
    """
    This program simulates ballistic transportation along x-axis.
    """
    logger = logging.getLogger('Solver_log')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('RGF_solver.log',mode='w')
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    ############################################################
    # Environment setup
    # 1. build up "output" folder.
    # 2. get user's inputs. Input type using sys.argv.
    ############################################################
    logger.info('========Start Solver========')
    RGF_parser = RGF_solver()
    setup_dict, job_dict, sweep_dict = RGF_parser.load_inputs()
    logger.info('========Import complete========')
    ############################################################
    # Run simulation
    # 0. Create POR
    # 1. Create splits of a single job
    # 2. Generate unit cell
    # 3. Calculate band diagram
    # 4. Calculate RGF
    ############################################################
    if setup_dict['POR enable']:
        '''
        Calculate POR
        '''
        for job_name, job in job_dict.items():
            ## make directory
            RGF_parser.job_dir = RGF_parser.output_dir+job_name
            RGF_parser.job_name = job_name
            if not os.path.exists(RGF_parser.job_dir): os.mkdir(RGF_parser.job_dir)
            '''
            Generate unit cell
            '''
            unit_list = RGF_parser.gen_unitCell(setup_dict, job)
            if setup_dict['POR Band structure']:
                '''
                Calculate band structure
                '''
                RGF_parser.cal_bandStructure(setup_dict, unit_list)
            if setup_dict['POR RGF']:
                '''
                Calculate RGF
                '''
                for key, unit in unit_list.items():
                    if unit.region['E_idx'][0] == None: unit.region['E_idx'] = range(0,len(eig[0][1]),1)
                    if unit.region['S_idx'][0] == None: unit.region['S_idx'] = range(0,int(setup_dict['mesh']),1)
                    E_list = unit.region['E_idx']
                    S_list = unit.region['S_idx']
                else:
                    split_summary = {'POR':[]}
                    CB_cache, split_summary = RGF_parser.cal_RGF_transmission(setup_dict, unit_list, E_list, S_list, split_summary, 'POR')
    else:
        logger.warning('Skip structure check')
    if setup_dict['Split enable']:
        '''
        Create splits
        '''
        if sweep_dict == {}:
            split_table = RGF_parser.create_splits(job)
            '''
            Calculate splits
            '''
            split_summary = {}
            for s_idx, split in enumerate(split_table):
                logger.info("Calculating split: "+str(s_idx))
                split_summary[s_idx] = []
                ## resolve calculation condition
                # get kx list
                kx_list = RGF_parser.resolve_mesh(split['kx'])
                RGF_parser.kx_list = kx_list
                # get band list
                CB_list = RGF_parser.resolve_mesh(split['CB'])
                RGF_parser.CB_list = CB_list
                '''
                Generate unit cell
                '''
                unit_list = RGF_parser.gen_unitCell(setup_dict)
                '''
                Calculate band diagram
                '''
                RGF_parser.cal_bandStructure(setup_dict, unit_list)
                '''
                Calculate RGF
                '''
                CB_cache, split_summary = RGF_parser.cal_RGF_transmission(setup_dict, unit_list, split_summary)
                '''
                Calculate time-dependent strucutre
                '''
                
                logger.info('Split_'+str(s_idx)+' complete!!\n')
            else:
                '''
                Summary table
                '''
                RGF_parser.gen_summary(setup_dict, CB_cache, split_summary)
        else:
            '''
            Calculate splits
            '''
            split_table = RGF_parser.create_splits_from_dict(sweep_dict)
            split_summary = {}
            for key, val in split_table.items():
                logging.info("Calculating split: "+key)
                split_summary[key] = []
                if setup_dict['Band diagram']:
                    '''
                    Generate unit cell
                    '''
                    unit_list = RGF_parser.gen_unitCell(setup_dict, job)
                    '''
                    Calculate band diagram
                    '''
                    RGF_parser.cal_bandStructure(setup_dict, unit_list)
                if setup_dict['Magnetic moment']:
                    '''
                    Generate unit cell
                    '''
                    unit_list = RGF_parser.gen_unitCell(setup_dict, job, option='_magnetic')
                    '''
                    Calculate magnetic moment
                    '''
                    RGF_parser.cal_bandStructure(setup_dict, unit_list)
                if setup_dict['RGF']:
                    '''
                    Calculate RGF
                    '''
                    CB_cache, split_summary = RGF_parser.cal_RGF_transmission(setup_dict, unit_list, split_summary)
                if setup_dict['TDGF']:
                    '''
                    Calculate RGF
                    '''
                    pass

                '''
                Calculate time-dependent strucutre
                '''
                
                logger.info('Split_'+key+' complete!!\n')
            else:
                '''
                Summary table
                '''
                #RGF_parser.gen_summary(setup_dict, CB_cache, split_summary)
                pass

    else:
        logger.info('Total time: '+str(round(RGF_parser.t_total,3))+' (sec)')
        logger.info('Program finished successfully')