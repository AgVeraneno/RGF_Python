import copy, os, time
from multiprocessing import Pool
import numpy as np
import data_util

class TD():
    def __init__(self, unit):
        self.cpu_turbo = 8
        self.unit = unit
        self.hq = self.unit.mat.q/self.unit.mat.h_bar
        self.h_bar = self.unit.mat.h_bar
        self.m_size = self.unit.m_size
        self.dt = 0.01                  # fs
        self.t0 = 0                     # start time (fs)
        self.tn = 1                   # stop time (fs)
        self.record_step = 1
        self.t_mesh = np.arange(self.t0,self.tn,self.dt)
    def initial_state(self):
        H = self.unit.H
        Pf = self.unit.Pf
        Pb = self.unit.Pb
        Gnx = []
        z_mat = np.zeros((self.m_size,self.m_size), dtype=np.complex128)
        I_mat = np.eye(self.m_size, dtype=np.complex128)
        for l in range(self.unit.L):
            Gnx.append(z_mat)
        else:
            t_const = -1j*self.dt*1e-15*self.hq
            Gnx[-3] = t_const**2*np.dot(Pf,np.dot(H,Pb))
            Gnx[-2] = t_const*Pf+t_const**2*(np.dot(Pf,H)+np.dot(H,Pf))
            Gnx[-1] = I_mat+t_const*H + t_const**2*(I_mat+np.dot(H,H))
            return np.block(Gnx)
    def cal_TDRG(self, Gnx):
        Gnx_tmp = copy.deepcopy(Gnx)
        for m in range(self.unit.L):
            Gxn = self.calU(m)
            Gnx_tmp[:,self.m_size*m:self.m_size*(m+1)] = np.dot(Gnx,Gxn)
        else:
            return Gnx_tmp
    def calU(self, m_idx):
        z_mat = np.zeros((self.m_size,self.m_size), dtype=np.complex128)
        I_mat = np.eye(self.m_size, dtype=np.complex128)
        Gxn = []
        for l in range(self.unit.L):
            Gxn.append(z_mat)
        else:
            t_const = -1j*self.dt*1e-15*self.hq
            if m_idx == 0:
                Gxn[m_idx] = I_mat+t_const*self.unit.H
                Gxn[m_idx+1] = t_const*self.unit.Pf
            elif m_idx == len(Gxn)-1:
                Gxn[m_idx-1] = t_const*self.unit.Pb
                Gxn[m_idx] = I_mat+t_const*self.unit.H
            else:
                Gxn[m_idx-1] = t_const*self.unit.Pb
                Gxn[m_idx] = I_mat+t_const*self.unit.H
                Gxn[m_idx+1] = t_const*self.unit.Pf
            return np.transpose(np.block(Gxn))
if __name__ == '__main__':
    import unit_cell_graphene, lib_material
    from matplotlib import pyplot as mplot
    test_mat = lib_material.Material('Graphene')
    test_job = {'width': [1],
                'length': [5],
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
    i_state = np.zeros((sum(test_struct.W)*test_struct.L,1))
    i_state[:sum(test_struct.W)] = 1/sum(test_struct.W)**0.5
    TD_parser.i_state = i_state
    T = []
    t_list = []
    for t_idx, t in enumerate(TD_parser.t_mesh):
        if t_idx == 0:
            Gnx = TD_parser.initial_state()
        else:
            Gnx = TD_parser.cal_TDRG(Gnx)
        o_state = np.dot(Gnx, i_state)
        if t_idx%TD_parser.record_step == 0:
            T.append(np.vdot(o_state, o_state))
            t_list.append(t*1e-15)
        else:
            continue
    mplot.plot(t_list, np.real(T))
    mplot.show()