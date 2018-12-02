import copy
import numpy as np
import band_structure as bs

class GreenFunction():
    def __init__(self, inputs, unit_list):
        self.inputs = inputs
        self.band_parser = bs.BandStructure(inputs)
        self.mat = inputs['material']
        self.unit_info = inputs['Unit cell']
        self.unit_list = unit_list
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
            else:
                E = copy.deepcopy(val[CB_idx])
            ## Initialize E matrix ##
            E_mat = np.eye(np.size(self.unit_list[0].H,0), dtype=np.complex128)*E
            P_phase = np.exp(-1j*unit.kx*self.mat.ax)-np.exp(1j*unit.kx*self.mat.ax)
            if u_ptr == 0:
                ## Generate first Green matrix: G00 ##
                Gnn = np.linalg.inv(E_mat-unit.H-unit.P_minus*np.exp(1j*unit.kx*self.mat.ax))
                Gn0 = copy.deepcopy(Gnn)
                ## Calculate reflection matrix ##
                self.R_matrix += np.dot(Gnn, unit.P_minus*P_phase)
                self.J0 = 1j*self.mat.ax/self.mat.h_bar*\
                          (unit.P_plus*np.exp(1j*unit.kx*self.mat.ax)-\
                           unit.P_minus*np.exp(-1j*unit.kx*self.mat.ax))
            elif u_ptr == len(self.mesh_grid)-1:
                ## Calculate last Gnn ##
                G_inv = E_mat - unit.H\
                        -unit.P_plus*np.exp(1j*unit.kx*self.mat.ax)\
                        -np.dot(unit.P_minus,np.dot(Gnn,unit.P_plus))
                Gnn = np.linalg.inv(G_inv)
                ## Calculate Gn0 ##
                Gn0 = np.dot(Gnn, np.dot(unit.P_minus,Gn0))
            else:
                ## Calculate Gnn ##
                G_inv = E_mat - unit.H\
                        -np.dot(unit.P_minus,np.dot(Gnn,unit.P_plus))
                Gnn = np.linalg.inv(G_inv)
                ## Calculate Gn0 ##
                Gn0 = np.dot(Gnn, np.dot(unit.P_minus,Gn0))
        self.T_matrix = np.dot(Gn0,unit.P_minus*P_phase)
        self.Jn = 1j*self.mat.ax/self.mat.h_bar*\
                  (unit.P_plus*np.exp(1j*unit.kx*self.mat.ax)-\
                   unit.P_minus*np.exp(-1j*unit.kx*self.mat.ax))
    def calTR(self, i_state):
        ## change format to cupy ##
        i_state = np.asarray(i_state)
        ## calculate states ##
        c0 = i_state
        cn = np.dot(self.T_matrix, i_state)
        c0_minus = np.dot(self.R_matrix, i_state)
        ## calculate current ##
        Ji = np.dot(np.conj(np.transpose(c0)), np.dot(self.J0, c0))
        Jt = np.dot(np.conj(np.transpose(cn)), np.dot(self.Jn, cn))
        Jr = np.dot(np.conj(np.transpose(c0_minus)), np.dot(self.J0, c0_minus))
        if abs(Ji/1.6e-19) >= 1e-5:
            T = Jt/Ji
            R = Jr/Ji
        else:
            T = R = 0
        return T,R
    def calRGF_GPU(self, E):
        ## Initialize E matrix and R matrix ##
        E_mat = cp.eye(np.size(self.unit_list[0].H,0), dtype=cp.complex128)*E
        self.R_matrix = cp.eye(np.size(self.unit_list[0].H,0), dtype=cp.complex128)*-1
        ## Calculate recursive Green's function ##
        for mesh_idx, u_ptr in enumerate(self.mesh_grid):
            '''
            mesh_idx: mesh index
            u_ptr: unit cell index 
            '''
            unit = self.unit_list[u_ptr]    # unit cell object
            unit.setKx(mesh_idx)
            P_phase = cp.exp(-1j*unit.kx*self.mat.ax)-cp.exp(1j*unit.kx*self.mat.ax)
            H_GPU = cp.asarray(unit.H).astype(cp.complex128)
            P_plus_GPU = cp.asarray(unit.P_plus).astype(cp.complex128)
            P_minus_GPU = cp.asarray(unit.P_minus).astype(cp.complex128)
            if mesh_idx == 0:
                ## Generate first Green matrix: G00 ##
                Ginv = E_mat-H_GPU-P_minus_GPU*cp.exp(1j*unit.kx*self.mat.ax)
                Gnn = np.linalg.inv(cp.asnumpy(Ginv))
                Gnn = cp.asarray(Gnn)
                Gn0 = copy.deepcopy(Gnn)
                ## Calculate reflection matrix ##
                self.R_matrix += cp.dot(Gnn, P_minus_GPU*P_phase)
                self.J0 = 1j*self.mat.ax/self.mat.h_bar*\
                          (P_plus_GPU*cp.exp(1j*unit.kx*self.mat.ax)-\
                           P_minus_GPU*cp.exp(-1j*unit.kx*self.mat.ax))
            elif mesh_idx == len(self.mesh_grid)-1:
                ## Calculate last Gnn ##
                G_inv = E_mat - H_GPU\
                        -P_plus_GPU*cp.exp(1j*unit.kx*self.mat.ax)\
                        -cp.dot(P_minus_GPU,cp.dot(Gnn,P_plus_GPU))
                Gnn = cp.linalg.pinv(G_inv)
                ## Calculate Gn0 ##
                Gn0 = cp.dot(Gnn, cp.dot(P_minus_GPU,Gn0))
            else:
                ## Calculate Gnn ##
                Ginv = E_mat-H_GPU-cp.dot(P_minus_GPU,cp.dot(Gnn,P_plus_GPU))
                Gnn = np.linalg.inv(cp.asnumpy(Ginv))
                Gnn = cp.asarray(Gnn)
                ## Calculate Gn0 ##
                Gn0 = cp.dot(Gnn, cp.dot(P_minus_GPU,Gn0))
        self.T_matrix = cp.dot(Gn0,P_minus_GPU*P_phase)
        self.Jn = 1j*self.mat.ax/self.mat.h_bar*\
                  (P_plus_GPU*cp.exp(1j*unit.kx*self.mat.ax)-\
                   P_minus_GPU*cp.exp(-1j*unit.kx*self.mat.ax))
    def calTR_GPU(self, i_state):
        ## change format to cupy ##
        i_state = cp.asarray(i_state)
        ## calculate states ##
        c0 = i_state
        cn = cp.dot(self.T_matrix, i_state)
        c0_minus = cp.dot(self.R_matrix, i_state)
        ## calculate current ##
        Ji = cp.dot(cp.conj(cp.transpose(c0)), cp.dot(self.J0, c0))
        Jt = cp.dot(cp.conj(cp.transpose(cn)), cp.dot(self.Jn, cn))
        Jr = cp.dot(cp.conj(cp.transpose(c0_minus)), cp.dot(self.J0, c0_minus))
        if abs(Ji/1.6e-19) >= 1e-5:
            T = Jt/Ji
            R = Jr/Ji
        else:
            T = R = 0
        return T,R