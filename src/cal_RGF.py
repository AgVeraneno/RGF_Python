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
        kx_list = []
        '''
        get incident state and energy
        '''
        band_parser = cal_band.CPU(self.setup, unit)
        kx, val, vec = band_parser.calState(kx_idx, True)
        E = val[self.CB]
        i_state = vec[:,self.CB]
        kx_list.append(kx)
        '''
        derive kx of other band with same energy
        '''
        kx2_idx = 0
        E2 = val[self.CB+1]
        E_pre = 0
        kx2 = kx
        kx_pre = None
        while True:
            if kx2_idx > kx_idx:
                break
            elif not np.isclose(E,E2):
                E_pre = copy.deepcopy(E2)
                kx_pre = copy.deepcopy(kx2)
                kx2, val, _ = band_parser.calState(kx2_idx, True)
                E2 = val[self.CB+1]
                kx2_idx += 1
            else:
                kx_list.append(kx2)
                break
            ## apply insert method if E2 has crossed E
            if E_pre < E and E2 > E:
                kx2_idx = (E2-E)/(E2-E_pre)
                kx2, val, _ = band_parser.calState(kx2_idx, True)
                kx_list.append(kx2)
                break
        return kx_list, E, i_state
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
        kx_list, E, i_state = self.setBand(input_unit, kx_idx)
        m_size = np.size(input_unit.H,0)
        E_matrix = np.eye(m_size, dtype=np.complex128)*np.real(E)
        Ji_list = []
        Jt_list = []
        ## calculate RGF ##
        phase_p = []
        phase_n = []
        P_phase = []
        KT_list = []
        T_list = []
        for kx in kx_list:
            ## phase terms
            phase_p.append(np.exp(1j*kx*self.mat.ax))
            phase_n.append(np.exp(-1j*kx*self.mat.ax))
            P_phase.append(phase_n[-1]-phase_p[-1])
        for mesh_idx, key in enumerate(mesh_grid):
            unit = self.unit_list[key]
            H = unit.H
            Pp = unit.Pf
            Pn = unit.Pb
            if self.reflect:
                '''
                Reflection
                '''
                if mesh_idx == 0:
                    ## Calculate first G00 and Gnn
                    G_inv = E_matrix - H - Pn*phase_p[0]
                    Gnn = np.linalg.inv(G_inv)
                elif mesh_idx == len(mesh_grid)-1:
                    ## Calculate last Gnn and Gn0
                    G_inv = E_matrix - H - Pp*phase_p[0] - np.matmul(Pn, np.matmul(Gnn,Pp))
                    Gnn = np.linalg.inv(G_inv)
                else:
                    ## Calculate Gnn and Gn0
                    G_inv = E_matrix - H - np.matmul(Pn, np.dot(Gnn,Pp))
                    Gnn = np.linalg.inv(G_inv)
            else:
                '''
                Transmission
                '''
                if mesh_idx == 0:
                    ## Calculate lead input Green's function
                    G_inv = E_matrix - H - Pp*phase_p[0]
                    Gnn = np.linalg.inv(G_inv)
                    Gn0 = copy.deepcopy(Gnn)
                elif mesh_idx == len(mesh_grid)-1:
                    ## Calculate last Gnn and Gn0
                    G_inv = E_matrix - H - Pn*phase_p[0] - np.matmul(Pp, np.matmul(Gnn,Pn))
                    Gnn = np.linalg.inv(G_inv)
                    Gn0 = np.matmul(Gnn, np.matmul(Pp,Gn0))
                    for i in range(len(kx_list)):
                        KT_list.append((E_matrix - H)*phase_p[i] + Pn*phase_p[i]**2+Pp)
                else:
                    ## Calculate Gnn and Gn0
                    G_inv = E_matrix - H - np.matmul(Pp, np.matmul(Gnn,Pn))
                    Gnn = np.linalg.inv(G_inv)
                    Gn0 = np.matmul(Gnn, np.matmul(Pp,Gn0))
        else:
            ## calculate T
            J0 = 1j*self.mat.ax/self.mat.h_bar*(Pn*phase_p[0]-Pp*phase_n[0])
            if self.reflect:
                T_matrix = np.eye(m_size, dtype=np.complex128)*-1 + np.dot(Gnn, np.matmul(Pp,P_phase[0]))
            else:
                T_matrix = np.matmul(Gn0, Pp*P_phase[0])
            for i in range(len(kx_list)):
                try:
                    new_Tmat = -np.matmul(np.matmul(np.linalg.inv(KT_list[i%2]-KT_list[(i+1)%2]),KT_list[(i+1)%2]),T_matrix)
                    T, Jt, Ji = self.calTR(i_state, new_Tmat, J0)
                    T_list.append(T)
                except:
                    T_list.append(0)
            t_mesh_stop = time.time() - t_mesh_start
            print('Mesh point @ kx=',str(kx_idx),' time:',t_mesh_stop, ' (sec)')
            return kx_list[0]*self.mat.ax/np.pi, E, T_list[0]
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
        return T, Jt, Ji
    def sort_E(self, table):
        output = copy.deepcopy(table)
        E_sort = np.argsort(table[:,0], axis=0)
        for i, E_idx in enumerate(E_sort):
            output[i, :] = np.array(table)[E_idx, :]
        return output