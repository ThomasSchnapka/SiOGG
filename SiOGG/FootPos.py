# -*- coding: utf-8 -*-
"""
construction of optimization variable w_p:
    w_p = [w_s^1, ..., w_s^{n_s}]
    w_p^s = [p^s_1, ..., p^s_{n_f}]
    p^s_f = [p^s_f_x, p^s_f_y]
"""


import numpy as np

class FootPos:
    '''
    class for easier access to leg position optimization variable w_p.
    
    Meant to be constructed every time there is a new w
    '''
    def __init__(self, problem, w_p):
        self.problem = problem
        self.n_s = problem.n_s     # number of steps
        self.n_f = problem.n_f     # number of feet
        #self.n_u = problem.n_u     # number of lambda (vertex weights) PER STEP
        #self.n_w_u = problem.n_w_u  # number of total lambda values
        #self.n_w_c = problem.n_w_c  # number of com optimization variables
        self.n_w_p = problem.n_w_p
        self.w_p = w_p
        assert(len(w_p)==problem.n_w_p)
        
        
    def get_foot_pos(self, k_s, dim):
        '''return position of all n_f feet in step k in dimension dim'''
        #pos = self.w[self.n_f*2*k:self.n_f*2*(k+1)]
        pos = self.w_p[2*self.n_f*k_s:2*self.n_f*(k_s+1)]
        if dim=="x":
            return pos[0::2]
        elif dim=="y":
            return pos[1::2]
        else:
            raise ValueError(f"there is no such dim {dim}")
            
            
    def grad_get_foot_pos(self, k_s, dim):
        '''gradient of get_feet_pos wrt w_p
        returns gradient as well as array containing indices of nonzero elements'''
        grad = np.zeros((self.n_w_p))
        if dim=="x":
            offset = 0
        elif dim=="y":
            offset = 1
        else:
            raise ValueError(f"there is no such dim {dim}")
        idx = np.zeros(self.n_f, dtype=int)
        for i_f in range(self.n_f):
            idx[i_f] = 2*self.n_f*k_s + i_f*2 + offset
            grad[idx[i_f]] = 1
        return grad, idx
