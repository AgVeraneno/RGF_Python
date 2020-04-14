import copy, os, time
from multiprocessing import Pool
import numpy as np
import data_util

class TD():
    def __init__(self, unit):
        self.unit = unit
        self.hq = self.unit.mat.q/self.unit.mat.h_bar
        self.h_bar = self.unit.mat.h_bar
        self.m_size = self.unit.m_size
        self.dt = 0.001                  # fs
        self.t0 = 0                     # start time (fs)
        self.tn = 2                   # stop time (fs)
        self.t_mesh = np.arange(self.t0,self.tn,self.dt)
    def __cpu__(self, t_idx):
        o_state = 0
        for tt in range(t_idx):
            if tt == 0:
                Gnx = self.calU(tt)
            else:
                Gnx = self.cal_TDRG(self.t_mesh[t_idx-tt], Gnx)
        else:
            if t_idx == 0:
                return 0
            else:
                o_state = np.dot(Gnx, self.i_state)
                return np.vdot(o_state, o_state)
    def cal_TDRG(self, t_idx, Gnx):
        Gnx_tmp = copy.deepcopy(Gnx)
        for m in range(self.unit.L):
            Gxn = self.calU(t_idx, m)
            Gnx_tmp[:,self.m_size*m:self.m_size*(m+1)] = np.dot(Gnx,Gxn)
        else:
            return Gnx_tmp
    def calU(self, t_idx, m_idx=0):
        z_mat = np.zeros((self.m_size,self.m_size), dtype=np.complex128)
        I_mat = np.eye(self.m_size, dtype=np.complex128)
        if t_idx == 0:
            Gnx = []
            for l in range(self.unit.L):
                Gnx.append(z_mat)
            else:
                Gnx[-2] = -1j*self.dt*1e-15*self.hq*self.unit.Pf
                Gnx[-1] = I_mat-1j*self.dt*1e-15*self.hq*self.unit.H
                return np.block(Gnx)
        else:
            Gxn = []
            for l in range(self.unit.L):
                Gxn.append(z_mat)
            else:
                if m_idx == 0:
                    Gxn[m_idx] = I_mat-1j*self.dt*1e-15*self.hq*self.unit.H
                    Gxn[m_idx+1] = -1j*self.dt*1e-15*self.hq*self.unit.Pf
                elif m_idx == len(Gxn)-1:
                    Gxn[m_idx-1] = -1j*self.dt*1e-15*self.hq*self.unit.Pb
                    Gxn[m_idx] = I_mat-1j*self.dt*1e-15*self.hq*self.unit.H
                else:
                    Gxn[m_idx-1] = -1j*self.dt*1e-15*self.hq*self.unit.Pb
                    Gxn[m_idx] = I_mat-1j*self.dt*1e-15*self.hq*self.unit.H
                    Gxn[m_idx+1] = -1j*self.dt*1e-15*self.hq*self.unit.Pf
                return np.transpose(np.block(Gxn))
if __name__ == '__main__':
    import unit_cell_graphene, lib_material
    from matplotlib import pyplot as mplot
    test_mat = lib_material.Material('Graphene')
    test_job = {'width': [1],
                'length': [3],
                'gap': [0],
                'Vtop':[0],
                'Vbot':[0]}
    test_setup = {'material':test_mat,
                  'kx_mesh':1001}

        
    test_struct = unit_cell_graphene.test(test_setup, test_job)
    TD_parser = TD(test_struct)
    '''
    test input
    '''
    i_state = np.zeros((test_struct.L,1))
    i_state[0][0] = 1
    TD_parser.i_state = i_state
    T = []
    t_len = len(TD_parser.t_mesh)
    with Pool(processes=20) as mcore:
        T = mcore.map(TD_parser.__cpu__, range(t_len))
    mplot.plot(TD_parser.t_mesh*1e-15, np.real(T))
    mplot.show()