import copy, os
import numpy as np
import band_structure as bs
from matplotlib import pyplot
import lib_excel, blas

class GreenFunction():
    def __init__(self, inputs, unit_list):
        self.inputs = inputs
        self.band_parser = bs.BandStructure(inputs)
        self.mat = inputs['material']
        self.unit_info = inputs['Unit cell']
        self.unit_list = unit_list
        self.E = []
        self.T = []
        self.R = []
        self.__meshing__()
    def __meshing__(self):
        """
        create a list with unit cell vs mesh
        """
        self.mesh_grid = []
        for u_idx, unit in enumerate(self.unit_info):
            counter = 0
            while counter < unit['L']:
                self.mesh_grid.append(u_idx)
                counter += 1
    def plotTR(self):
        ## construct vectors
        try:
            pyplot.plot(np.array(self.E)/1.6e-19,np.array(self.T))
        except:
            pyplot.plot(cp.asnumpy(self.E)/1.6e-19,cp.asnumpy(self.T))
        pyplot.ylim([0,1])
        pyplot.xlabel("E (eV)")
        pyplot.ylabel("T")
        ## output to figures
        thisUnit = self.inputs['Unit cell'][0]
        if self.inputs['material'].name == 'Graphene' \
        and self.inputs['direction'] == 'Armchair':
            filename = 'T_'+self.inputs['lattice']+'_AGNR_'
        elif self.inputs['material'].name == 'Graphene' \
        and self.inputs['direction'] == 'Zigzag':
            filename = 'T_'+self.inputs['lattice']+'_ZGNR_'
        condition = 'Z='+str(thisUnit['Region'])+\
                    ',Type='+str(thisUnit['Type'])+\
                    ',S='+str(thisUnit['Shift'])+\
                    ',W='+str(thisUnit['W'])+\
                    ',L='+str(thisUnit['L'])+\
                    ',Vtop='+str(thisUnit['Vtop'])+\
                    ',Vbot='+str(thisUnit['Vbot'])+\
                    ',d='+str(thisUnit['delta'])
        try:
            pyplot.savefig('figures/'+filename+condition+'.png')
        except:
            os.mkdir('figures')
            pyplot.savefig('figures/'+filename+condition+'.png')
        pyplot.close()
    def saveAsXLS(self):
        thisUnit = self.inputs['Unit cell'][0]
        if self.inputs['material'].name == 'Graphene' \
        and self.inputs['direction'] == 'Armchair':
            filename = 'TR_'+self.inputs['lattice']+'_AGNR_'
        elif self.inputs['material'].name == 'Graphene' \
        and self.inputs['direction'] == 'Zigzag':
            filename = 'TR_'+self.inputs['lattice']+'_ZGNR_'
        condition = 'Z='+str(thisUnit['Region'])+\
                    ',Type='+str(thisUnit['Type'])+\
                    ',S='+str(thisUnit['Shift'])+\
                    ',W='+str(thisUnit['W'])+\
                    ',L='+str(thisUnit['L'])+\
                    ',Vtop='+str(thisUnit['Vtop'])+\
                    ',Vbot='+str(thisUnit['Vbot'])+\
                    ',d='+str(thisUnit['delta'])
        excel_parser = lib_excel.excel('T&R/'+filename+condition+'.xlsx')
        ## create T sheet ##
        excel_parser.newWorkbook('T')
        for i in range(len(self.T)):
            _ = excel_parser.worksheet.cell(column=1, row=i+1,\
                                            value="=COMPLEX("+str(np.real(self.E[i]))\
                                                +","+str(np.imag(self.E[i]))+")")
            _ = excel_parser.worksheet.cell(column=2, row=i+1,\
                                            value="=COMPLEX("+str(np.real(self.T[i]))\
                                                +","+str(np.imag(self.T[i]))+")")
        ## create R sheet ##
        excel_parser.newSheet('R')
        for i in range(len(self.R)):
            _ = excel_parser.worksheet.cell(column=1, row=i+1,\
                                            value="=COMPLEX("+str(np.real(self.E[i]))\
                                                +","+str(np.imag(self.E[i]))+")")
            _ = excel_parser.worksheet.cell(column=2, row=i+1,\
                                            value="=COMPLEX("+str(np.real(self.R[i]))\
                                                +","+str(np.imag(self.R[i]))+")")
        try:
            excel_parser.save()
        except:
            os.mkdir('T&R')
            excel_parser.save()
    def calRGF(self, kx_idx):
        for mesh_idx, u_ptr in enumerate(self.mesh_grid):
            ## identify unit cell ##
            unit = self.unit_list[u_ptr]
            ## calculate eigenstate ##
            val, vec = self.band_parser.calState(unit, kx_idx)
            if mesh_idx == 0:
                CB_idx = self.band_parser.getCBidx(unit.info['delta'], val)
                E = copy.deepcopy(val[CB_idx])
                i_state = copy.deepcopy(vec[:,CB_idx])
                self.R_matrix = np.eye(np.size(self.unit_list[0].H,0), dtype=np.complex128)*-1
                self.E.append(E)
            else:
                E = copy.deepcopy(val[CB_idx])
            ## Initialize E matrix ##
            E_mat = np.eye(np.size(self.unit_list[0].H,0), dtype=np.complex128)*E
            P_phase = np.exp(-1j*unit.kx*self.mat.ax)-np.exp(1j*unit.kx*self.mat.ax)
            if mesh_idx == 0:
                ## Generate first Green matrix: G00 ##
                Gnn = np.linalg.inv(E_mat-unit.H-unit.P_minus*np.exp(1j*unit.kx*self.mat.ax))
                Gn0 = copy.deepcopy(Gnn)
                ## Calculate reflection matrix ##
                self.R_matrix += np.dot(Gnn, unit.P_minus*P_phase)
                self.J0 = 1j*self.mat.ax/self.mat.h_bar*\
                          (unit.P_plus*np.exp(1j*unit.kx*self.mat.ax)-\
                           unit.P_minus*np.exp(-1j*unit.kx*self.mat.ax))
            elif mesh_idx == len(self.mesh_grid)-1:
                ## Calculate last Gnn ##
                G_inv = E_mat - unit.H\
                        -unit.P_plus*np.exp(1j*unit.kx*self.mat.ax)
                Gnn = np.linalg.inv(G_inv)
                ## Calculate Gn0 ##
                Gn0 = np.dot(Gnn, np.dot(unit.P_minus,Gn0))
            else:
                ## Calculate Gnn ##
                G_inv = E_mat - unit.H-np.dot(unit.P_minus, np.dot(Gnn,unit.P_plus))
                Gnn = np.linalg.inv(G_inv)
                ## Calculate Gn0 ##
                Gn0 = np.dot(Gnn, np.dot(unit.P_minus,Gn0))
        self.T_matrix = np.dot(Gn0,unit.P_minus*P_phase)
    def calTR(self, i_state):
        ## change format to cupy ##
        i_state = np.asarray(i_state)
        ## calculate states ##
        c0 = i_state
        cn = np.dot(self.T_matrix, i_state)
        c0_minus = np.dot(self.R_matrix, i_state)
        ## calculate current ##
        Ji = np.dot(np.conj(np.transpose(c0)), np.dot(self.J0, c0))
        Jt = np.dot(np.conj(np.transpose(cn)), np.dot(self.J0, cn))
        Jr = np.dot(np.conj(np.transpose(c0_minus)), np.dot(self.J0, c0_minus))
        if abs(Ji/1.6e-19) >= 1e-5:
            T = Jt/Ji
            R = Jr/Ji
        else:
            T = R = 0
        self.T.append(T)
        self.R.append(R)
        return T,R
    def calRGFT_GPU(self, kx_idx, CB_idx):
        global cp
        import cupy as cp
        for mesh_idx, u_ptr in enumerate(self.mesh_grid):
            ## identify unit cell ##
            if mesh_idx > 0 and self.mesh_grid[mesh_idx-1] == u_ptr:
                pass
            else:
                unit = self.unit_list[u_ptr]
                H = cp.asarray(unit.H)
                P_plus = cp.asarray(unit.P_plus)
                P_minus = cp.asarray(unit.P_minus)
            ## calculate eigenstate ##
            val, vec = self.band_parser.calState(unit, kx_idx)
            if mesh_idx == 0:
                E = copy.deepcopy(val[CB_idx])
                self.i_state = copy.deepcopy(vec[:,CB_idx])      # Cn0
                self.R_matrix = cp.eye(np.size(self.unit_list[0].H,0), dtype=cp.complex128)*-1
                self.E.append(E)
            else:
                E = copy.deepcopy(val[CB_idx])
            ## Initialize E matrix ##
            E_mat = cp.eye(np.size(self.unit_list[0].H,0), dtype=cp.complex128)*E
            P_phase = P_minus*(cp.exp(-1j*unit.kx*self.mat.ax)-cp.exp(1j*unit.kx*self.mat.ax))
            if mesh_idx == 0:
                ## Generate first Green matrix: G00 ##
                G_inv = E_mat-H-P_minus*cp.exp(1j*unit.kx*self.mat.ax)
                Gnn = blas.inv(G_inv)
                Gn0 = copy.deepcopy(Gnn)
                ## Calculate reflection matrix ##
                self.R_matrix += cp.dot(Gnn, P_phase)
                self.J0 = 1j*self.mat.ax/self.mat.h_bar*\
                          (P_plus*cp.exp(1j*unit.kx*self.mat.ax)-\
                           P_minus*cp.exp(-1j*unit.kx*self.mat.ax))
            elif mesh_idx == len(self.mesh_grid)-1:
                ## Calculate last Gnn ##
                G_inv = E_mat - H -P_plus*cp.exp(1j*unit.kx*self.mat.ax)
                Gnn = blas.inv(G_inv)
                ## Calculate Gn0 ##
                Gn0 = cp.dot(Gnn, cp.dot(P_minus,Gn0))
            else:
                ## Calculate Gnn ##
                G_inv = E_mat - H-cp.dot(P_minus,cp.dot(Gnn,P_plus))
                Gnn = blas.inv(G_inv)
                ## Calculate Gn0 ##
                Gn0 = cp.dot(Gnn, cp.dot(P_minus,Gn0))
        self.T_matrix = cp.dot(Gn0,P_phase)
    def calRGFR_GPU(self, kx_idx, CB_idx):
        global cp
        import cupy as cp
        for mesh_idx_inv in range(len(self.mesh_grid)):
            mesh_idx = len(self.mesh_grid)-mesh_idx_inv-1
            u_ptr = self.mesh_grid[mesh_idx]
            ## identify unit cell ##
            if mesh_idx_inv > 0 and self.mesh_grid[mesh_idx+1] == u_ptr:
                pass
            else:
                unit = self.unit_list[u_ptr]
                H = cp.asarray(unit.H)
                P_plus = cp.asarray(unit.P_plus)
                P_minus = cp.asarray(unit.P_minus)
            ## calculate eigenstate ##
            val, vec = self.band_parser.calState(unit, kx_idx)
            if mesh_idx == 0:
                E = copy.deepcopy(val[CB_idx])
                self.i_state = copy.deepcopy(vec[:,CB_idx])      # Cn0
                self.R_matrix = cp.eye(np.size(self.unit_list[0].H,0), dtype=cp.complex128)*-1
            else:
                E = copy.deepcopy(val[CB_idx])
            ## Initialize E matrix ##
            E_mat = cp.eye(np.size(self.unit_list[0].H,0), dtype=cp.complex128)*E
            P_phase = P_minus*(cp.exp(-1j*unit.kx*self.mat.ax)-cp.exp(1j*unit.kx*self.mat.ax))
            if mesh_idx == 0:
                ## Generate first Green matrix: G00 ##
                G_inv = E_mat-H-P_minus*cp.exp(1j*unit.kx*self.mat.ax)
                Gnn = blas.inv(G_inv)
                ## Calculate reflection matrix ##
                self.R_matrix += cp.dot(Gnn, P_phase)
            elif mesh_idx == len(self.mesh_grid)-1:
                ## Calculate last Gnn ##
                G_inv = E_mat - H -P_plus*cp.exp(1j*unit.kx*self.mat.ax)
                Gnn = blas.inv(G_inv)
            else:
                ## Calculate Gnn ##
                G_inv = E_mat - H-cp.dot(P_minus,cp.dot(Gnn,P_plus))
                Gnn = blas.inv(G_inv)
    def calTR_GPU(self):
        ## change format to cupy ##
        i_state = cp.asarray(self.i_state)
        ## calculate states ##
        c0 = i_state
        cn = cp.dot(self.T_matrix, i_state)
        c0_minus = cp.dot(self.R_matrix, i_state)
        ## calculate current ##
        Ji = cp.dot(cp.conj(cp.transpose(c0)), cp.dot(self.J0, c0))
        Jt = cp.dot(cp.conj(cp.transpose(cn)), cp.dot(self.J0, cn))
        Jr = cp.dot(cp.conj(cp.transpose(c0_minus)), cp.dot(self.J0, c0_minus))
        if abs(Ji/1.6e-19) >= 1e-5:
            T = complex(Jt/Ji)
            R = complex(Jr/Ji)
        else:
            T = R = 0
        self.T.append(T)
        self.R.append(R)
        return T,R