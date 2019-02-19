import copy, os, time
import numpy as np
import data_util, cal_band

class CPU():
    def __init__(self, setup, unit_list):
        self.setup = setup
        self.mat = setup['material']
        self.unit_list = unit_list
        self.__meshing__(unit_list)
        self.reflect = False
        self.CB_idx = np.arange(int(setup['CB_idx_start'])-1,int(setup['CB_idx_stop']),1)
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
    def setBand(self, unit, kx_idx, CB_idx):
        band_parser = cal_band.CPU(self.setup, unit)
        kx, val, vec = band_parser.calState(kx_idx, True)
        E = val[CB_idx]
        i_state = vec[:,CB_idx]
        return kx, E, i_state
    def calRGF_transmit(self, kx_idx):
        Jt_tot = 0+0j
        Ji_tot = 0+0j
        if self.reflect:
            mesh_grid = copy.deepcopy(self.mesh_grid)
            mesh_grid.reverse()
        else:
            mesh_grid = self.mesh_grid
        t_mesh_start = time.time()
        for CB_idx in self.CB_idx:
            kx, E, i_state = self.setBand(self.unit_list[mesh_grid[0]], kx_idx, CB_idx)
            m_size = np.size(self.unit_list[mesh_grid[0]].H,0)
            E_matrix = np.eye(m_size, dtype=np.complex128)*E
            ## calculate RGF ##
            for mesh_idx, key in enumerate(mesh_grid):
                unit = self.unit_list[key]
                H = unit.H
                Pp = unit.P_plus
                Pn = unit.P_minus
                ## initialize all component for RGF
                if mesh_idx == 0:
                    ## phase terms
                    phase_p = np.exp(1j*kx*self.mat.ax)
                    phase_n = np.exp(-1j*kx*self.mat.ax)
                    P_phase = phase_n-phase_p
                    ## Generate first Green matrix: G00 ##
                    G_inv = (E_matrix-H)-Pn*phase_p
                    Gnn = np.linalg.inv(G_inv)
                    Gn0 = copy.deepcopy(Gnn)
                elif mesh_idx == len(self.mesh_grid)-1:
                    ## Calculate last Gnn ##
                    G_inv = E_matrix - H - Pp*phase_p - np.dot(Pn, np.dot(Gnn,Pp))
                    Gnn = np.linalg.inv(G_inv)
                    ## Calculate Gn0 ##
                    Gn0 = np.dot(Gnn, np.dot(Pn,Gn0))
                else:
                    ## Calculate Gnn ##
                    G_inv = E_matrix - H - np.dot(Pn, np.dot(Gnn,Pp))
                    Gnn = np.linalg.inv(G_inv)
                    ## Calculate Gn0 ##
                    Gn0 = np.dot(Gnn, np.dot(Pn,Gn0))
            else:
                T_matrix = np.dot(Gn0,Pn*P_phase)
                ## calculate T and R ##
                J0 = 1j*self.mat.ax/self.mat.h_bar*(Pp*phase_p-Pn*phase_n)
                JT, Ji, T = self.calTR(i_state, T_matrix, J0)
                Jt_tot += JT
                Ji_tot += Ji
                t_mesh_stop = time.time() - t_mesh_start
        print('Mesh point @ kx=',str(kx_idx),'CB=',str(CB_idx+1),' time:',t_mesh_stop, ' (sec)')
        return kx*self.mat.ax/np.pi, E, Jt_tot/Ji_tot, Jt_tot, Ji_tot
    def calTR(self, i_state, Tmat, J0):
        ## calculate states ##
        c0 = i_state
        cn = np.dot(Tmat, i_state)
        ## calculate current ##
        Ji = np.vdot(c0, np.dot(J0, c0))
        Jt = np.vdot(cn, np.dot(J0, cn))
        if not np.isclose(np.real(Ji),0):
            T = Jt/Ji
        else:
            T = 0
        return Jt, Ji, T
    def sort_E(self, table):
        output = copy.deepcopy(table)
        E_sort = np.argsort(table[:,0], axis=0)
        for i, E_idx in enumerate(E_sort):
            output[i, :] = np.array(table)[E_idx, :]
        return output