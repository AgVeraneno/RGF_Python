import sys, copy, csv
import lib_material, lib_excel, unit_cell, data_util
import numpy as np
#from matplotlib import pyplot

def load_setup(setup_file):
    setup = {}
    job = {}
    with open(setup_file, newline='') as csv_file:
        rows = csv.DictReader(csv_file)
        for row in rows:
            '''
            setup dictionary
            '''
            if row['setting'] == 'material':
                setup['material'] = lib_material.Material(row['value'])
            elif row['setting'] == 'lattice':
                setup['lattice'] = row['value']
            elif row['setting'] == 'direction':
                setup['direction'] = row['value']
            elif row['setting'] == 'structure':
                setup['structure'] = row['value']
            elif row['setting'] == 'kx_mesh':
                setup['kx_mesh'] = row['value']
            elif row['setting'] == 'debug':
                if row['value'] == 'TRUE':
                    setup['debug'] = True
                else:
                    setup['debug'] = False
            elif row['setting'] == 'SU_type':
                setup['SU_type'] = row['value']
            elif row['setting'] == 'band':
                if row['value'] == 'TRUE':
                    setup['band'] = True
                else:
                    setup['band'] = False
            elif row['setting'] == 'RGF':
                if row['value'] == 'TRUE':
                    setup['RGF'] = True
                else:
                    setup['RGF'] = False
            elif row['setting'] == 'SOI_open':
                if row['value'] == 'TRUE':
                    setup['SOI'] = True
                else:
                    setup['SOI'] = False
            elif row['setting'] == 'spin_header':
                setup['header'] = data_util.str2float1D(row['value'],totem=';',dtype='str')
                setup['spin'] = [data_util.str2float1D(setup['header'][0],dtype='str')[1],
                                 data_util.str2float1D(setup['header'][1],dtype='str')[1]]
            else:
                pass
            '''
            job dictionary
            '''
            if row['enable'] == 'o':
                if row['name'] not in job:      # new job. create new key entrance
                    job[row['name']] = {'name': row['name'],
                                        'kx': row['kx_mesh'],
                                        'CB': row['band'],
                                        'region list': [row['region']],
                                        row['region']: {}}
                    job[row['name']][row['region']]['shift'] = [int(row['shift'])]
                    job[row['name']][row['region']]['width'] = [int(row['width'])]
                    job[row['name']][row['region']]['length'] = [int(row['length'])]
                    job[row['name']][row['region']]['Vtop'] = [float(row['Vtop'])]
                    job[row['name']][row['region']]['Vbot'] = [float(row['Vbot'])]
                    job[row['name']][row['region']]['gap'] = [float(row['gap'])]
                    job[row['name']][row['region']]['Ez'] = [float(row['Ez'])]
                    job[row['name']][row['region']]['Bx'] = [float(row['Bx'])]
                    job[row['name']][row['region']]['By'] = [float(row['By'])]
                    job[row['name']][row['region']]['Bz'] = [float(row['Bz'])]
                    job[row['name']][row['region']]['sweep_var'] = [row['sweep_parameter']]
                    job[row['name']][row['region']]['sweep_val'] = [row['sweep_value']]
                else:
                    if row['region'] not in job[row['name']].keys():
                        job[row['name']]['region list'].append(row['region'])
                        job[row['name']][row['region']] = {}
                        job[row['name']][row['region']]['shift'] = [int(row['shift'])]
                        job[row['name']][row['region']]['width'] = [int(row['width'])]
                        job[row['name']][row['region']]['length'] = [int(row['length'])]
                        job[row['name']][row['region']]['Vtop'] = [float(row['Vtop'])]
                        job[row['name']][row['region']]['Vbot'] = [float(row['Vbot'])]
                        job[row['name']][row['region']]['gap'] = [float(row['gap'])]
                        job[row['name']][row['region']]['Ez'] = [float(row['Ez'])]
                        job[row['name']][row['region']]['Bx'] = [float(row['Bx'])]
                        job[row['name']][row['region']]['By'] = [float(row['By'])]
                        job[row['name']][row['region']]['Bz'] = [float(row['Bz'])]
                        job[row['name']][row['region']]['sweep_var'] = [row['sweep_parameter']]
                        job[row['name']][row['region']]['sweep_val'] = [row['sweep_value']]
                    else:
                        job[row['name']][row['region']]['shift'].append(int(row['shift']))
                        job[row['name']][row['region']]['width'].append(int(row['width']))
                        job[row['name']][row['region']]['length'].append(int(row['length']))
                        job[row['name']][row['region']]['Vtop'].append(float(row['Vtop']))
                        job[row['name']][row['region']]['Vbot'].append(float(row['Vbot']))
                        job[row['name']][row['region']]['gap'].append(float(row['gap']))
                        job[row['name']][row['region']]['Ez'].append(float(row['Ez']))
                        job[row['name']][row['region']]['Bx'].append(float(row['Bx']))
                        job[row['name']][row['region']]['By'].append(float(row['By']))
                        job[row['name']][row['region']]['Bz'].append(float(row['Bz']))
                        job[row['name']][row['region']]['sweep_var'].append(row['sweep_parameter'])
                        job[row['name']][row['region']]['sweep_val'].append(row['sweep_value'])
            else:
                pass
    return setup, job, {}
def importFromCSV(setup_file, job_file):
    setup = {'isDebug':False,
             'isGPU':False,
             'isReflect':False,
             'isParallel':False,
             'parallel_CPU':1,
             'material':None,
             'lattice':None,
             'direction':None,
             'brief':None,
             'SU_type':None,
             'SU_hopping_size':None,
             'brick_size':None,
             'kx_mesh':None,
             'mesh_start':0,
             'mesh_stop':0,
             'isPlot_band':False,
             'isPlot_zoom':False,
             'isRGF':False,
             'CB_idx_start':None,
             'CB_idx_stop':None}
    job = {'name':None,
           'region':-1,
           'cell_type':1,
           'shift':0,
           'width':0,
           'length':0,
           'Vtop':0,
           'Vbot':0,
           'gap':0}
    '''
    import setup
    '''
    with open(setup_file,newline='') as csv_file:
        rows = csv.DictReader(csv_file)
        for row in rows:
            for key in setup.keys():
                if key[0:2] == 'is':
                    if row[key] == '1':
                        setup[key] = True
                    elif row[key] == '0':
                        setup[key] = False
                    else:
                        raise ValueError('Incorrect input in job file:', row[key])
                elif key == 'material':
                    setup[key] = lib_material.Material(row[key])
                else:
                    setup[key] = row[key]
    '''
    import jobs
    '''
    with open(job_file,newline='') as csv_file:
        rows = csv.DictReader(csv_file)
        job_list = []
        for row in rows:
            if row['enable'] == 'o':
                new_job = copy.deepcopy(job)
                for key in job.keys():
                    new_job[key] = row[key]
                job_list.append(new_job)
            else:
                continue
    return setup, job_list
def saveAsCSV(file_name, table):
    with open(file_name, 'w', newline='') as csv_file:
        csv_parser = csv.writer(csv_file, delimiter=',')
        for i in range(np.size(np.array(table), 0)):
            try:
                csv_parser.writerow(list(table[i,:]))
            except:
                csv_parser.writerow(table[i])

def importFromExcel(filename=None):
    with lib_excel.excel(file=filename) as excel_parser:
        '''
        Load __setup__ sheet
        '''
        setup = {}
        for row in excel_parser.readSheet('__setup__'):
            if row[0].value == 'Value':
                setup['Material'] = row[1].value
                setup['Lattice'] = row[2].value
                setup['Direction'] = row[3].value
                setup['mesh'] = int(row[4].value)
            else:
                continue
        '''
        Load structure sheet
        '''
        structure = {}
        for row in excel_parser.readSheet('structure'):
            if row[0].value == 'o':
                ## create new region
                this_region = {}
                this_region['Job'] = row[1].value
                this_region['Name'] = row[2].value
                this_region['Width'] = [int(row[3].value)]
                this_region['Length'] = int(row[4].value)
                this_region['Vdrop'] = [row[5].value]
                this_region['Vtop'] = float(row[6].value)
                this_region['Vbot'] = float(row[7].value)
                this_region['gap'] = [float(row[8].value)]
                this_region['E'] = {}
                this_region['E']['z'] = [float(row[9].value)]
                this_region['B'] = {}
                this_region['B']['x'] = [float(row[10].value)]
                this_region['B']['y'] = [float(row[11].value)]
                this_region['B']['z'] = [float(row[12].value)]
                if row[1].value not in structure: structure[row[1].value] = {}
                structure[row[1].value][row[2].value] = this_region
            elif row[0].value == '>':
                this_region = structure[row[1].value][row[2].value]
                this_region['Width'].append(int(row[3].value))
                this_region['Vdrop'].append(row[5].value)
                this_region['gap'].append(float(row[8].value))
                this_region['E']['z'].append(float(row[9].value))
                this_region['B']['x'].append(float(row[10].value))
                this_region['B']['y'].append(float(row[11].value))
                this_region['B']['z'].append(float(row[12].value))
            else:
                continue
        '''
        Load sweep sheet
        '''
        sweep = {}
        for row in excel_parser.readSheet('sweep'):
            if row[0].value == 'o':
                ## create new sweep
                this_sweep = {}
                this_sweep['Name'] = row[1].value
                this_sweep['Ref_job'] = row[2].value
                this_sweep['Sweep_list'] = []
                for i in range(int((len(row)-3)/3)):
                    new_split = {}
                    new_split['Region'] = row[3+3*i].value
                    new_split['var'] = row[4+3*i].value
                    new_split['val'] = row[5+3*i].value
                    this_sweep['Sweep_list'].append(new_split)
                else:
                    sweep[row[1].value] = this_sweep
            else:
                continue
    return setup, structure, sweep
'''
def saveAsExcel(inputs, u_idx, unit, input_array=None, save_type=None):
    lattice = inputs['lattice']
    mat = inputs['material'].name
    dir = inputs['direction']
    file_name = str(u_idx)+"_"+save_type+"_"+lattice+"_"+dir[0]+mat[0]+"NR.xlsx"
    excel_parser = lib_excel.excel('../output/'+file_name)
    condition = 'Z='+str(unit.info['Region'])+\
                ',Type='+str(unit.info['Type'])+\
                ',S='+str(unit.info['Shift'])+\
                ',W='+str(unit.info['W'])+\
                ',L='+str(unit.info['L'])+\
                ',Vtop='+str(unit.info['Vtop'])+\
                ',Vbot='+str(unit.info['Vbot'])+\
                ',d='+str(unit.info['delta'])
    if save_type == 'matrix':
        ## create H sheet ##
        excel_parser.newWorkbook('H')
        for i in range(np.size(unit.H, 0)):
            for j in range(np.size(unit.H, 1)):
                _ = excel_parser.worksheet.cell(column=j+1, row=i+1,\
                                                value="=COMPLEX("+str(np.real(unit.H[i,j]))\
                                                +","+str(np.imag(unit.H[i,j]))+")")
        ## create P sheet ##
        excel_parser.newSheet('P')
        P = unit.P_plus+unit.P_minus
        for i in range(np.size(unit.H, 0)):
            for j in range(np.size(unit.H, 1)):
                _ = excel_parser.worksheet.cell(column=j+1, row=i+1,\
                                                value="=COMPLEX("+str(np.real(P[i,j]))\
                                                +","+str(np.imag(P[i,j]))+")")
        excel_parser.save()
    elif save_type == 'TR':
        ## create T sheet ##
        excel_parser.newWorkbook('T')
        for i in range(len(input_array['T'])):
            _ = excel_parser.worksheet.cell(column=1, row=i+1,\
                                            value=str(np.real(input_array['E'][i])))
            _ = excel_parser.worksheet.cell(column=2, row=i+1,\
                                            value=str(np.real(input_array['T'][i])))
        ## create R sheet ##
        excel_parser.newSheet('R')
        for i in range(len(input_array['R'])):
            _ = excel_parser.worksheet.cell(column=1, row=i+1,\
                                            value=str(np.real(input_array['E'][i])))
            _ = excel_parser.worksheet.cell(column=2, row=i+1,\
                                            value=str(np.real(input_array['R'][i])))
        excel_parser.save()
'''
def saveAsFigure(setup, u_idx, unit, table, save_type=None):
    from matplotlib import pyplot
    lattice = setup['lattice']
    mat = setup['material'].name
    dir = setup['direction']
    file_name = str(u_idx)+"_"+save_type+"_"+"lead"
    if save_type == 'band':
        ## construct vectors
        eig_mat = np.array(table)[:,1:]
        kx_sweep = np.array(table)[:,0]
        for y_idx in range(np.size(eig_mat,1)):
            pyplot.plot(kx_sweep,eig_mat[:,y_idx])
            #pyplot.xlim([0,1])
            pyplot.ylim([-10,10])
            pyplot.xlabel("kx*ax/pi")
            pyplot.ylabel("E (eV)")
        ## output to figures

        pyplot.savefig('../output/'+file_name+'.png')
        pyplot.close()
        ## plot zoom in figure if enabled
        if setup['isPlot_zoom']:
            for y_idx in range(np.size(eig_mat,1)):
                pyplot.plot(kx_sweep,eig_mat[:,y_idx])
                pyplot.xlim([0,0.4])
                pyplot.ylim([-0.5,0.5])
                pyplot.xlabel("kx*ax/pi")
                pyplot.ylabel("E (eV)")
            pyplot.savefig('../output/'+file_name+'_zoom.png')
            pyplot.close()
    elif save_type == 'TR':
        E = input_array['E']
        T = input_array['T']
        R = input_array['R']
        pyplot.plot(E, T)
        pyplot.ylim([-0.05,1.05])
        pyplot.xlabel("E (eV)")
        pyplot.ylabel("T")
        pyplot.savefig('../output/'+file_name+condition+'.png')
        pyplot.close()