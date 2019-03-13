import copy, os, time
import numpy as np
from scipy import linalg as LA
import data_util, cal_band

class CPU():
    def __init__(self, setup, unit_list):
        self.setup = setup
        self.mat = setup['material']
        self.unit_list = unit_list
        self.__meshing__(unit_list)
        self.reflect = False
        self.CB = 0
    def __meshing__(self, unit_list):
        """
        create a list with unit cell vs mesh
        """
        self.mesh_grid = []
        for key, zone in unit_list.items():
            counter = 0
            while counter < zone.L:
                self.mesh_grid.append(key)
                counter += 1
    def setBand(self, unit, kx_idx):
        '''
        get incident state and energy
        '''
        band_parser = cal_band.CPU(self.setup, unit)
        kx, val, vec = band_parser.calState(kx_idx, True)
        E = val[self.CB]
        i_state = vec[:,self.CB]
        return kx, E, i_state
    def calRGF_transmit(self, kx_idx):
        t_mesh_start = time.time()
        '''
        calculate in forward or inverse mode 
        '''
        if self.reflect:
            mesh_grid = copy.deepcopy(self.mesh_grid)
            mesh_grid.reverse()
        else:
            mesh_grid = self.mesh_grid
        '''
        calculate RGF with assigned conduction band
        '''
        ## calculate incident state
        input_unit = self.unit_list[mesh_grid[0]]
        kx, E, i_state = self.setBand(input_unit, kx_idx)
        m_size = np.size(input_unit.H,0)
        E_matrix = np.eye(m_size, dtype=np.complex128)*np.real(E)
        ## phase terms
        phase_p = np.exp(1j*kx*self.mat.ax)
        phase_n = np.exp(-1j*kx*self.mat.ax)
        P_phase = phase_n-phase_p
        ## calculate RGF ##
        for mesh_idx, key in enumerate(mesh_grid):
            unit = self.unit_list[key]
            H = unit.H
            Pp = unit.Pf
            Pn = unit.Pb
            if self.reflect:
                if mesh_idx == 0:
                    ## Calculate first G00 and Gnn
                    G_inv = E_matrix - H - Pn*phase_p
                    Gnn = np.linalg.inv(G_inv)
                elif mesh_idx == len(mesh_grid)-1:
                    ## Calculate last Gnn and Gn0
                    G_inv = E_matrix - H - Pp*phase_p - np.matmul(Pn, np.matmul(Gnn,Pp))
                    Gnn = np.linalg.inv(G_inv)
                else:
                    ## Calculate Gnn and Gn0
                    G_inv = E_matrix - H - np.matmul(Pn, np.dot(Gnn,Pp))
                    Gnn = np.linalg.inv(G_inv)
            else:
                if mesh_idx == 0:
                    ## Calculate lead input Green's function
                    G_inv = E_matrix - H + Pp*phase_p
                    Gnn = np.linalg.inv(G_inv)
                    Gn0 = copy.deepcopy(Gnn)
                elif mesh_idx == len(mesh_grid)-1:
                    ## Calculate last Gnn and Gn0
                    G_inv = E_matrix - H + Pn*phase_p - np.matmul(Pn, np.matmul(Gnn,Pp))
                    Gnn = np.linalg.inv(G_inv)
                    Gn0 = np.matmul(Gnn, np.matmul(Pn,Gn0))
                else:
                    ## Calculate Gnn and Gn0
                    G_inv = E_matrix - H - np.matmul(Pn, np.matmul(Gnn,Pp))
                    Gnn = np.linalg.inv(G_inv)
                    Gn0 = np.matmul(Gnn, np.matmul(Pn,Gn0))
        else:
            ## calculate T
            J0 = 1j*self.mat.ax/self.mat.h_bar*(Pn*phase_p-Pp*phase_n)
            if self.reflect:
                T_matrix = np.eye(m_size, dtype=np.complex128)*-1 + np.dot(Gnn, np.matmul(Pp,P_phase))
            else:
                T_matrix = -np.matmul(Gn0, Pp*P_phase)
            T = self.calTR(i_state, T_matrix, J0)
            t_mesh_stop = time.time() - t_mesh_start
            print('Mesh point @ kx=',str(kx_idx),' time:',t_mesh_stop, ' (sec)')
            return kx*self.mat.ax/np.pi, E, T
    def calTR(self, i_state, Tmat, J0):
        ## calculate states ##
        c0 = i_state
        cn = np.dot(Tmat, i_state)
        ## calculate current ##
        Ji = np.vdot(c0, np.matmul(J0, c0))
        Jt = np.vdot(cn, np.matmul(J0, cn))
        if not np.isclose(np.real(Ji),0):
            T = Jt/Ji
        else:
            T = 0
        return T
    def sort_E(self, table):
        output = copy.deepcopy(table)
        E_sort = np.argsort(table[:,0], axis=0)
        for i, E_idx in enumerate(E_sort):
            output[i, :] = np.array(table)[E_idx, :]
        return output