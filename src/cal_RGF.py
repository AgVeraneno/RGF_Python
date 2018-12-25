import copy, os, time
import numpy as np
import cal_band
from matplotlib import pyplot
#import lib_excel, blas

class CPU():
    def __init__(self, inputs, unit_list):
        self.inputs = inputs
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
    def setBand(self, inputs, unit):
        self.band_parser = cal_band.CPU(inputs, unit)
    def calRGF_transmit(self, kx_idx):
        t_mesh_start = time.time()
        ## calculate RGF ##
        for mesh_idx, u_idx in enumerate(self.mesh_grid):
            unit = self.unit_list[u_idx]
            unit.setKx(kx_idx)
            ## initialize all component for RGF
            if mesh_idx == 0:
                ## calculate incident state ##
                self.setBand(self.inputs, unit)
                val, vec = self.band_parser.calState(kx_idx)
                CB_idx = self.band_parser.getCBidx(unit.info['delta'], val)
                E = copy.deepcopy(val[CB_idx])
                E_matrix = np.eye(np.size(self.unit_list[0].H,0), dtype=np.complex128)*E
                i_state = copy.deepcopy(vec[:,CB_idx])
                # reimplement H, P+ and p-
                H = unit.H
                Pp = unit.P_plus
                Pn = unit.P_minus
                P_phase = np.exp(-1j*unit.kx*self.mat.ax)-np.exp(1j*unit.kx*self.mat.ax)
                # initialize R matrix
                R_matrix = np.eye(np.size(self.unit_list[0].H,0), dtype=np.complex128)*-1
                ## Generate first Green matrix: G00 ##
                G_inv = E_matrix - H - Pn*np.exp(1j*unit.kx*self.mat.ax)
                Gnn = np.linalg.inv(G_inv)
                Gn0 = copy.deepcopy(Gnn)
                J0 = 1j*1.6e-19*self.mat.ax/self.mat.h_bar*\
                        (Pp*np.exp(1j*unit.kx*self.mat.ax)-\
                        Pn*np.exp(-1j*unit.kx*self.mat.ax))
            elif mesh_idx == len(self.mesh_grid)-1:
                ## Calculate last Gnn ##
                G_inv = E_matrix - H - Pp*np.exp(1j*unit.kx*self.mat.ax) - np.dot(Pn, np.dot(Gnn,Pp))
                Gnn = np.linalg.inv(G_inv)
                ## Calculate Gn0 ##
                Gn0 = np.dot(Gnn, np.dot(Pn,Gn0))
            else:
                ## Calculate Gnn ##
                G_inv = E_matrix - H - np.dot(Pn, np.dot(Gnn,Pp))
                Gnn = np.linalg.inv(G_inv)
                ## Calculate Gn0 ##
                Gn0 = np.dot(Gnn, np.dot(Pn,Gn0))
        T_matrix = np.dot(Gn0,Pn*P_phase)
        ## calculate T and R ##
        T, R = self.calTR(i_state, T_matrix, R_matrix, J0)
        t_mesh_stop = time.time() - t_mesh_start
        print('Mesh point @',str(kx_idx),' time:',t_mesh_stop, ' (sec)')
        return E,T,R
    def calRGF_reflect(self, kx_idx):
        ## calculate RGF ##
        mesh_grid_r = copy.deepcopy(self.mesh_grid)
        mesh_grid_r.reverse()
        for mesh_idx, u_idx in enumerate(mesh_grid_r):
            unit = self.unit_list[u_idx]
            ## initialize all component for RGF
            if mesh_idx == 0:
                ## calculate incident state ##
                val, vec = self.band_parser.calState(kx_idx)
                CB_idx = self.band_parser.getCBidx(unit.info['delta'], val)
                E = copy.deepcopy(val[CB_idx])
                E_matrix = np.eye(np.size(self.unit_list[0].H,0), dtype=np.complex128)*E
                i_state = copy.deepcopy(vec[:,CB_idx])
                # reimplement H, P+ and p-
                H = unit.H
                Pp = unit.P_plus
                Pn = unit.P_minus
                P_phase = np.exp(-1j*unit.kx*self.mat.ax)-np.exp(1j*unit.kx*self.mat.ax)
                # initialize R matrix
                R_matrix = np.eye(np.size(self.unit_list[0].H,0), dtype=np.complex128)*-1
                ## Generate first Green matrix: G00 ##
                G_inv = E_matrix - H - Pn*np.exp(1j*unit.kx*self.mat.ax)
                Gnn = np.linalg.inv(G_inv)
                Gn0 = copy.deepcopy(Gnn)
                J0 = 1j*1.6e-19*self.mat.ax/self.mat.h_bar*\
                          (Pp*np.exp(1j*unit.kx*self.mat.ax)-\
                           Pn*np.exp(-1j*unit.kx*self.mat.ax))
            elif mesh_idx == len(mesh_grid_r)-1:
                ## Calculate last Gnn ##
                G_inv = E_matrix - H - Pp*np.exp(1j*unit.kx*self.mat.ax) - np.dot(Pn, np.dot(Gnn,Pp))
                Gnn = np.linalg.inv(G_inv)
                ## Calculate Gn0 ##
                Gn0 = np.dot(Gnn, np.dot(Pn,Gn0))
            else:
                ## Calculate Gnn ##
                G_inv = E_matrix - H - np.dot(Pn, np.dot(Gnn,Pp))
                Gnn = np.linalg.inv(G_inv)
                ## Calculate Gn0 ##
                Gn0 = np.dot(Gnn, np.dot(Pn,Gn0))
        T_matrix = np.dot(Gn0,Pn*P_phase)
        T, R = self.calTR(i_state, T_matrix, R_matrix, J0)
        return E,T,R
    def calTR(self, i_state, Tmat, Rmat, J0):
        ## calculate states ##
        c0 = i_state
        cn = np.dot(Tmat, i_state)
        c0_minus = np.dot(Rmat, i_state)
        ## calculate current ##
        Ji = np.dot(np.conj(np.transpose(c0)), np.dot(J0, c0))
        Jt = np.dot(np.conj(np.transpose(cn)), np.dot(J0, cn))
        Jr = np.dot(np.conj(np.transpose(c0_minus)), np.dot(J0, c0_minus))
        if not np.isclose(np.real(Ji),0):
            T = np.abs(np.real(Jt/Ji))
            R = np.abs(np.real(Jr/Ji))
        else:
            T = R = 0
        return T,R