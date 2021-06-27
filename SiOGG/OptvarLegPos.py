# -*- coding: utf-8 -*-
"""
construction of optimization variable w_p:
    w_p = [w_s_1, ..., w_s_{n_s}]
    w_p^s = [p^s_1, ..., p^s_{n_f}]
    p^s_f = [p^s_f_x, p^s_f_y]
"""


import numpy as np

class OptvarLegPos:
    '''class for easier access to leg position optimization variable w_p'''
    def __init__(self, w, problem):
        self.w = w
        self.problem = problem
        self.n_s = problem.n_s     # number of steps
        self.n_f = problem.n_f     # number of feet
        self.n_u = problem.n_u     # number of lambda (vertex weights) PER STEP
        self.n_w_u = problem.n_w_u  # number of total lambda values
        self.n_w_c = problem.n_w_c  # number of com optimization variables
        
        self.w_p = w[self.n_w_c:self.n_w_u]
        
    def get_feet_pos(self, l_s, dim):
        '''return position of all n_f feet in step i_s in dimension dim'''
        pos = self.w[self.n_f*2*l_s:self.n_f*2*(l_s+1)]
        if dim=="x":
            return pos[0::2]
        elif dim=="y":
            return pos[1::2]
        else:
            raise ValueError(f"there is no such dim {dim}")
