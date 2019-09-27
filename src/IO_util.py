import sys, copy, csv
import lib_material, unit_cell
import numpy as np
from matplotlib import pyplot

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
                setup['debug'] = bool(row['value'])
            elif row['setting'] == 'SU_type':
                setup['SU_type'] = row['value']
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
                        job[row['name']][row['region']]['sweep_var'] = [row['sweep_parameter']]
                        job[row['name']][row['region']]['sweep_val'] = [row['sweep_value']]
                    else:
                        job[row['name']][row['region']]['shift'].append(int(row['shift']))
                        job[row['name']][row['region']]['width'].append(int(row['width']))
                        job[row['name']][row['region']]['length'].append(int(row['length']))
                        job[row['name']][row['region']]['Vtop'].append(float(row['Vtop']))
                        job[row['name']][row['region']]['Vbot'].append(float(row['Vbot']))
                        job[row['name']][row['region']]['gap'].append(float(row['gap']))
                        job[row['name']][row['region']]['sweep_var'].append(row['sweep_parameter'])
                        job[row['name']][row['region']]['sweep_val'].append(row['sweep_value'])
            else:
                pass
    return setup, job
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
        csv_parser = csv.writer(csv_file, delimiter=',',quotechar='|')
        for i in range(np.size(np.array(table), 0)):
            try:
                csv_parser.writerow(list(table[i,:]))
            except:
                csv_parser.writerow(table[i])
'''
def importFromExcel(filename=None):
    inputs = {'material': None,
              'lattice': None,
              'direction': None,
              'mesh': [0, 0],
              'kx_mesh': 0,
              'Vbias': [0.0, 0.0],
              'Unit cell': [],
              'function':{'isPlotBand':False,
                          'isPlotZoom':False},
              'CPU':{'p_enable':False,
                     'p_num':1},
              'GPU':{'enable': False,
                     'Max matrix':0}}
    with lib_excel.excel(filename) as excel_parser:
        for row in excel_parser.readSheet('__setup__'):
            if row[0].value == 'Using GPU':
                inputs['GPU']['enable'] = bool(row[1].value)
                inputs['GPU']['Max matrix'] = int(row[2].value)
            elif row[0].value == 'Using Parallel':
                inputs['CPU']['p_enable'] = bool(row[1].value)
                inputs['CPU']['p_num'] = int(row[2].value)
            elif row[0].value == 'Material':
                if str(row[1].value) == 'Graphene':
                    inputs['material'] = lib_material.Graphene()
            elif row[0].value == 'Lattice':
                inputs['lattice'] = str(row[1].value)
            elif row[0].value == 'Direction':
                inputs['direction'] = str(row[1].value)
            elif row[0].value == 'kx mesh':
                inputs['kx_mesh'] = int(row[1].value)
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
    return inputs

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
    lattice = setup['lattice']
    mat = setup['material'].name
    dir = setup['direction']
    file_name = str(u_idx)+"_"+save_type+"_"+setup['brief']+"_"+"lead"
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