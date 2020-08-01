import copy, os, time
import numpy as np
import data_util, cal_band

class CPU():
    def __init__(self, setup, unit_list):
        self.setup = setup
        self.mat = setup['Material']
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
    def setBand(self, unit, kx_idx, o_zone=False):
        kx_list = []
        '''
        get incident state and energy
        '''
        band_parser = cal_band.CPU(self.setup, unit)
        kx, val, vec = band_parser.calState(kx_idx, True)
        E = val[self.CB]
        i_state = vec[:,self.CB]
        kx_list.append(kx)
        if o_zone:
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
        output_unit = self.unit_list[mesh_grid[-1]]
        kx, E, i_state = self.setBand(input_unit, kx_idx-1)
        kx_list_o, _, _ = self.setBand(output_unit, kx_idx-1, True)
        m_size = np.size(input_unit.H,0)
        E_matrix = np.eye(m_size, dtype=np.complex128)*np.real(E)
        ## calculate RGF ##
        phase_p = np.exp(1j*kx[0]*self.mat.ax)
        phase_n = np.exp(-1j*kx[0]*self.mat.ax)
        P_phase = phase_n-phase_p
        phase_o = []
        for kx_o in kx_list_o:
            phase_o.append(np.exp(1j*kx_o*self.mat.ax))
        for mesh_idx, key in enumerate(mesh_grid):
            unit = self.unit_list[key]
            H = unit.H
            Pp = unit.Pb
            Pn = unit.Pf
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
                    T_matrix = np.eye(m_size, dtype=np.complex128)*-1 + np.dot(Gnn, np.matmul(Pp,P_phase[0]))
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
                    G_inv = E_matrix - H - Pp*phase_p
                    Gnn = np.linalg.inv(G_inv)
                    Gn0 = copy.deepcopy(Gnn)
                elif mesh_idx == len(mesh_grid)-1:
                    ## Calculate last Gnn and Gn0
                    G_inv = E_matrix - H - Pn*phase_p - np.matmul(Pp, np.matmul(Gnn,Pn))
                    Gnn = np.linalg.inv(G_inv)
                    Gn0 = np.matmul(Gnn, np.matmul(Pp,Gn0))
                    T_matrix = np.matmul(Gn0, Pp*P_phase)
                    CN = np.matmul(T_matrix, i_state)
                else:
                    ## Calculate Gnn and Gn0
                    G_inv = E_matrix - H - np.matmul(Pp, np.matmul(Gnn,Pn))
                    Gnn = np.linalg.inv(G_inv)
                    Gn0 = np.matmul(Gnn, np.matmul(Pp,Gn0))
        else:
            self.C0.append(i_state)
            self.CN.append(CN)
            ## Calculate multiple states
            G_inv = E_matrix - H - Pn*sum(phase_o)
            Gnn = np.linalg.inv(G_inv)
            CN_next = -np.dot(Gnn, np.dot(Pp-Pn*np.prod(phase_o),CN))
            J0 = 1j*self.mat.ax/self.mat.h_bar*(Pn*phase_p-Pp*phase_n)
            ## calculate T with spin include
            N = len(CN)
            CN1 = copy.deepcopy(CN)
            CN2 = copy.deepcopy(CN)
            i_state1 = copy.deepcopy(i_state)
            i_state2 = copy.deepcopy(i_state)
            for i in range(N):
                if i%2 == 0:
                    CN2[i] = 0
                    i_state2[i] = 0
                else:
                    CN1[i] = 0
                    i_state1[i] = 0
            ## calculate T
            Ji = np.vdot(i_state, np.matmul(J0, i_state))
            Jt1 = np.vdot(CN1, np.matmul(J0, CN1))
            Jt2 = np.vdot(CN2, np.matmul(J0, CN2))
            Jt3 = np.vdot(CN, np.matmul(J0, CN))
            T1 = self.calTR(i_state, CN1, J0)
            T2 = self.calTR(i_state, CN2, J0)
            T3 = self.calTR(i_state, CN, J0)
            t_mesh_stop = time.time() - t_mesh_start
            #print('Mesh point @ kx=',str(kx_idx),' time:',t_mesh_stop, ' (sec)')
            return kx[0]*self.mat.a, E, Jt1, Jt2, Jt3, Ji
    def calTR(self, i_state, o_state, J0):
        Ji = np.vdot(i_state, np.matmul(J0, i_state))
        Jt = np.vdot(o_state, np.matmul(J0, o_state))
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