import copy
import numpy as np

class GreenFunction():
    def __init__(self, inputs, unit_list):
        self.inputs = inputs
        self.mat = inputs['material']
        self.unit_info = inputs['Unit cell']
        self.unit_list = unit_list
        self.__meshing__()
        if inputs['GPU']['enable']:
            global cp
            import cupy as cp
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
    def calRGF(self, E):
        E_mat = np.eye(np.size(self.unit_list[0].H,0), dtype=np.complex128)*E
        self.R_matrix = np.eye(np.size(self.unit_list[0].H,0), dtype=np.complex128)*-1
        for u_ptr, n in enumerate(self.mesh_grid):
            unit = self.unit_list[n]    # unit cell object
            unit.setKx(u_ptr)
            P_phase = np.exp(-1j*unit.kx*self.mat.ax)-\
                      np.exp(1j*unit.kx*self.mat.ax)
            if u_ptr == 0:
                ## Generate first Green matrix: G00 ##
                Gnn = np.linalg.inv(E_mat-unit.H\
                                    -unit.P_minus*np.exp(1j*unit.kx*self.mat.ax))
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
    def calTR(self):
        Ji_matrix = np.dot(np.conj(np.transpose(self.c0)), np.dot(self.J0, self.c0))
        Jt_matrix = np.dot(np.conj(np.transpose(self.cn)), np.dot(self.Jn, self.cn))
        Jr_matrix = np.dot(np.conj(np.transpose(self.c0_minus)), np.dot(self.J0, self.c0_minus))
        if round(Ji_matrix, 25) != 0:
            T = Jt_matrix/Ji_matrix
            R = Jr_matrix/Ji_matrix
        else:
            T = R = 0
        return T,R
    def calState(self, i_state, o_state):
        self.c0 = i_state
        self.cn = o_state
        self.c0_minus = np.dot(self.R_matrix, i_state)
    def calRGF_GPU(self, E):
        E_mat = cp.eye(np.size(self.unit_list[0].H,0), dtype=cp.complex128)*E
        self.R_matrix = cp.eye(np.size(self.unit_list[0].H,0), dtype=cp.complex128)*-1
        for u_ptr, n in enumerate(self.mesh_grid):
            unit = self.unit_list[n]    # unit cell object
            unit.setKx(u_ptr)
            P_phase = cp.exp(-1j*unit.kx*self.mat.ax)-\
                      cp.exp(1j*unit.kx*self.mat.ax)
            H_GPU = cp.array(unit.H)
            P_plus_GPU = cp.array(unit.P_plus)
            P_minus_GPU = cp.array(unit.P_minus)
            if u_ptr == 0:
                ## Generate first Green matrix: G00 ##
                Gnn = cp.linalg.inv(E_mat-H_GPU\
                                    -P_minus_GPU*cp.exp(1j*unit.kx*self.mat.ax))
                Gn0 = copy.deepcopy(Gnn)
                ## Calculate reflection matrix ##
                self.R_matrix += cp.dot(Gnn, P_minus_GPU*P_phase)
                self.J0 = 1j*self.mat.ax/self.mat.h_bar*\
                          (P_plus_GPU*cp.exp(1j*unit.kx*self.mat.ax)-\
                           P_minus_GPU*cp.exp(-1j*unit.kx*self.mat.ax))
            elif u_ptr == len(self.mesh_grid)-1:
                ## Calculate last Gnn ##
                G_inv = E_mat - H_GPU\
                        -P_plus_GPU*cp.exp(1j*unit.kx*self.mat.ax)\
                        -cp.dot(P_minus_GPU,cp.dot(Gnn,P_plus_GPU))
                Gnn = cp.linalg.inv(G_inv)
                ## Calculate Gn0 ##
                Gn0 = cp.dot(Gnn, cp.dot(P_minus_GPU,Gn0))
            else:
                ## Calculate Gnn ##
                G_inv = E_mat - H_GPU\
                        -cp.dot(P_minus_GPU,cp.dot(Gnn,P_plus_GPU))
                Gnn = cp.linalg.inv(G_inv)
                ## Calculate Gn0 ##
                Gn0 = cp.dot(Gnn, cp.dot(P_minus_GPU,Gn0))
        self.T_matrix = cp.dot(Gn0,P_minus_GPU*P_phase)
        self.Jn = 1j*self.mat.ax/self.mat.h_bar*\
                  (P_plus_GPU*cp.exp(1j*unit.kx*self.mat.ax)-\
                   P_minus_GPU*cp.exp(-1j*unit.kx*self.mat.ax))
    def calTR_GPU(self):
        Ji_matrix = cp.dot(cp.conj(cp.transpose(self.c0)), cp.dot(self.J0, self.c0))
        Jt_matrix = cp.dot(cp.conj(cp.transpose(self.cn)), cp.dot(self.Jn, self.cn))
        Jr_matrix = cp.dot(cp.conj(cp.transpose(self.c0_minus)), cp.dot(self.J0, self.c0_minus))
        if abs(Ji_matrix/1.6e-19) >= 1e-5:
            T = Jt_matrix/Ji_matrix
            R = Jr_matrix/Ji_matrix
        else:
            T = R = 0
        return T,R
    def calState_GPU(self, i_state, o_state):
        self.c0 = i_state
        self.cn = o_state
        self.c0_minus = cp.dot(self.R_matrix, i_state)