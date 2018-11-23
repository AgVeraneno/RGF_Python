import copy
import numpy as np

class GreenFunction():
    def __init__(self, inputs, unit_list):
        self.inputs = inputs
        self.mat = inputs['material']
        self.unit_info = inputs['Unit cell']
        self.E = (inputs['Vbias'][0] - inputs['Vbias'][1])*1.6e-19
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
    def calRGF(self):
        E_mat = np.eye(np.size(self.unit_list[0].H,0), dtype=np.complex128)*self.E
        self.R_matrix = np.eye(np.size(self.unit_list[0].H,0), dtype=np.complex128)*-1
        for u_ptr, n in enumerate(self.mesh_grid):
            unit = self.unit_list[n]    # unit cell object
            unit.setKx(u_ptr)
            P_phase = np.exp(-1j*unit.kx*self.mat.ax)-\
                      np.exp(1j*unit.kx*self.mat.ax)
            if u_ptr == 0:
                ## Generate first Green matrix - G00 ##
                Gnn = np.linalg.inv(E_mat-unit.H + unit.P_minus*\
                                    np.exp(1j*unit.kx*self.mat.ax))
                Gn0 = copy.deepcopy(Gnn)
                ## Calculate reflection matrix ##
                self.R_matrix += np.dot(Gnn, unit.P_minus*P_phase)
                self.J0 = 1j*self.mat.ax/self.mat.h_bar*\
                          (unit.P_plus*np.exp(1j*unit.kx*self.mat.kx)-\
                           unit.P_minus*np.exp(-1j*unit.kx*self.mat.kx))
            else:
                ## Calculate Gnn ##
                G_inv = E_mat-unit.H -\
                        np.dot(unit.P_minus,np.dot(Gnn,unit.P_plus))
                Gnn = np.linalg.inv(G_inv)
                ## Calculate Gn0 ##
                Gn0 = np.dot(Gnn, np.dot(unit.P_minus,Gn0))
        self.T_matrix = np.dot(Gn0,unit.P_minus*P_phase)
        self.Jn = 1j*self.mat.ax/self.mat.h_bar*\
                  (unit.P_plus*np.exp(1j*unit.kx*self.mat.kx)-\
                   unit.P_minus*np.exp(-1j*unit.kx*self.mat.kx))
    def calTR(self, c0, cn, c0_minus):
        Ji = np.dot(np.transpose(c0), np.dot(self.J0, c0))
        Jt = np.dot(np.transpose(cn), np.dot(self.Jn, cn))
        Jr = np.dot(np.transpose(c0_minus), np.dot(self.J0, c0_minus))
        return Jt/Ji, Jr/Ji
        