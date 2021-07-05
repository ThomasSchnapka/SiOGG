# -*- coding: utf-8 -*-

import numpy as np


class CostFunction:
    '''
    CostFunction
    '''
    def __init__(self, problem):
        self.cont_seq = problem.cont_seq
        self.n_f = problem.n_f
        self.n_c = problem.n_c
        self.n_w_u = problem.n_w_u
        self.n_optvar = problem.n_optvar
        
        
    def cost(self, optvar):
        '''return cost associated with optimization variable w'''
        w_u = optvar.w_u
        w_u_opt = (  np.tile(np.ravel(self.cont_seq), 3) 
                   / np.tile(np.sum(self.cont_seq, axis=1), self.n_f*3))
        J = (w_u - w_u_opt)@(w_u - w_u_opt)
        
        # experimental: penalize high accelerations of COP:
        for i_c in range(self.n_c):
            J += np.square(optvar.cop.get_cop(i_c, 1, "x") - optvar.cop.get_cop(i_c, 0, "x"))
            J += np.square(optvar.cop.get_cop(i_c, 2, "x") - optvar.cop.get_cop(i_c, 1, "x"))
            J += np.square(optvar.cop.get_cop(i_c, 1, "y") - optvar.cop.get_cop(i_c, 0, "y"))
            J += np.square(optvar.cop.get_cop(i_c, 2, "y") - optvar.cop.get_cop(i_c, 1, "y"))
            
        return J
    
    
    def gradient(self, optvar):
        '''return gradient of cost associated with w'''
        w_u = optvar.w_u
        w_u_opt = (  np.tile(np.ravel(self.cont_seq), 3) 
                   / np.tile(np.sum(self.cont_seq, axis=1), self.n_f*3))
        # create empty gradient vector and fill it
        dw = np.zeros(self.n_optvar)
        dw[-self.n_w_u:] = 2*w_u - 2*w_u_opt
        #return np.zeros(self.n_optvar)
        #print(dw.shape, dw)
        return dw

    
    
    