import lib_excel, lib_material
import numpy as np
from matplotlib import pyplot

def importFromExcel(filename=None):
    inputs = {'material': None,
              'lattice': None,
              'direction': None,
              'mesh': [0, 0],
              'kx_mesh': 383,
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
def saveAsFigure(inputs, u_idx, unit, input_array, save_type=None):
    lattice = inputs['lattice']
    mat = inputs['material'].name
    dir = inputs['direction']
    file_name = str(u_idx)+"_"+save_type+"_"+lattice+"_"+dir[0]+mat[0]+"NR_"
    condition = 'Z='+str(unit.info['Region'])+\
                ',Type='+str(unit.info['Type'])+\
                ',S='+str(unit.info['Shift'])+\
                ',W='+str(unit.info['W'])+\
                ',L='+str(unit.info['L'])+\
                ',Vtop='+str(unit.info['Vtop'])+\
                ',Vbot='+str(unit.info['Vbot'])+\
                ',d='+str(unit.info['delta'])
    if save_type == 'band':
        ## construct vectors
        eig_mat = np.real(np.array(input_array['y']))
        kx_sweep = np.real(np.array(input_array['x']))
        for y_idx in range(np.size(eig_mat,1)):
            pyplot.plot(kx_sweep,eig_mat[:,y_idx])
            pyplot.xlim([0,1])
            pyplot.ylim([-10,10])
            pyplot.xlabel("kx*ax/pi")
            pyplot.ylabel("E (eV)")
        ## output to figures

        pyplot.savefig('../output/'+file_name+condition+'.png')
        pyplot.close()
        ## plot zoom in figure if enabled
        if inputs['function']['isPlotZoom']:
            for y_idx in range(np.size(eig_mat,1)):
                pyplot.plot(kx_sweep,eig_mat[:,y_idx])
                pyplot.xlim([0,1])
                pyplot.ylim([-1,1])
                pyplot.xlabel("kx*ax/pi")
                pyplot.ylabel("E (eV)")
            pyplot.savefig('../output/'+file_name+condition+'_zoom.png')
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