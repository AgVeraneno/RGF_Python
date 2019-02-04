import copy, os, time
import numpy as np
import data_util, cal_band

class CPU():
    def __init__(self, setup, unit_list):
        self.setup = setup
        self.mat = setup['material']
        self.unit_list = unit_list
        self.__meshing__(unit_list)
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
        band_parser = cal_band.CPU(self.setup, unit)
        kx, val, vec = band_parser.calState(kx_idx, True)
        CB_idx = unit.ML_size
        E = val[CB_idx]
        i_state = vec[:,CB_idx]
        ## remove i_state with energy is +-1000 (wall)
        wall = []
        for w in data_util.mfind(-1000, val):
            i_state[w] = 0
            wall.append(w)
        for w in data_util.mfind(1000, val):
            i_state[w] = 0
            wall.append(w)
        return kx, E, i_state, wall
    def calRGF_transmit(self, kx_idx):
        t_mesh_start = time.time()
        ## calculate RGF ##
        for mesh_idx, key in enumerate(self.mesh_grid):
            unit = self.unit_list[key]
            m_size = np.size(unit.H,0)
            H = unit.H
            Pp = unit.P_plus
            Pn = unit.P_minus
            ## initialize all component for RGF
            if mesh_idx == 0:
                ## calculate incident state ##
                kx, E, i_state, wall = self.setBand(unit, kx_idx)
                E_matrix = np.eye(m_size, dtype=np.complex128)*E
                phase_p = np.exp(1j*kx*self.mat.ax)
                phase_n = np.exp(-1j*kx*self.mat.ax)
                # re-implement H, P+ and p-
                P_phase = phase_n-phase_p
                # initialize R matrix
                R_matrix = np.eye(m_size, dtype=np.complex128)*-1
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
        T_matrix = np.dot(Gn0,Pn*P_phase)
        ## calculate T and R ##
        J0 = 1j*self.mat.ax/self.mat.h_bar*(Pp*phase_p-Pn*phase_n)
        T, R = self.calTR(i_state, T_matrix, R_matrix, J0, wall)
        t_mesh_stop = time.time() - t_mesh_start
        print('Mesh point @',str(kx_idx),' time:',t_mesh_stop, ' (sec)')
        return E,T,R
    def calRGF_reflect(self, kx_idx):
        ## calculate RGF ##
        mesh_grid_r = copy.deepcopy(self.mesh_grid)
        mesh_grid_r.reverse()
        for mesh_idx, u_idx in enumerate(mesh_grid_r):
            unit = self.unit_list[u_idx]
            H = unit.H
            Pp = unit.P_plus
            Pn = unit.P_minus
            ## initialize all component for RGF
            if mesh_idx == 0:
                ## calculate incident state ##
                val, vec = self.band_parser.calState(kx_idx)
                CB_idx = self.band_parser.getCBidx(unit.info['delta'], val)
                E = copy.deepcopy(val[CB_idx])
                E_matrix = np.eye(np.size(self.unit_list[0].H,0), dtype=np.complex128)*E
                i_state = copy.deepcopy(vec[:,CB_idx])
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
        return np.real(E),np.real(T),np.real(R)
    def calTR(self, i_state, Tmat, Rmat, J0, wall):
        ## calculate states ##
        c0 = i_state
        cn = np.dot(Tmat, i_state)
        c0_minus = np.dot(Rmat, i_state)
        ## remove wall
        for w in wall:
            cn[w] = 0
            c0_minus[w] = 0
        ## calculate current ##
        Ji = np.vdot(c0, np.dot(J0, c0))
        Jt = np.vdot(cn, np.dot(J0, cn))
        Jr = np.vdot(c0_minus, np.dot(J0, c0_minus))
        if not np.isclose(np.real(Ji),0):
            T = np.abs(np.real(Jt/Ji))
            R = np.abs(np.real(Jr/Ji))
        else:
            T = R = 0
        return T,R