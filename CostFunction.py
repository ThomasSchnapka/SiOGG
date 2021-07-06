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
        self.T_c = problem.T_c
        
        
    def cost(self, optvar):
        '''return cost associated with optimization variable w'''
        w_u = optvar.w_u
        w_u_opt = (  np.tile(np.ravel(self.cont_seq), 3) 
                   / np.tile(np.sum(self.cont_seq, axis=1), self.n_f*3))
        J = (w_u - w_u_opt)@(w_u - w_u_opt)
        
        # experimental: penalize high diffferences of COM:
        '''
        J += np.square(optvar.com.eval_spline(         0, "x", 0) - optvar.com.eval_spline(self.T_c/2, "x", 0))
        J += np.square(optvar.com.eval_spline(self.T_c/2, "x", 0) - optvar.com.eval_spline(  self.T_c, "x", 0))
        J += np.square(optvar.com.eval_spline(         0, "y", 0) - optvar.com.eval_spline(self.T_c/2, "y", 0))
        J += np.square(optvar.com.eval_spline(self.T_c/2, "y", 0) - optvar.com.eval_spline(  self.T_c, "y", 0))
        for i_c in range(1, self.n_c):
            J += np.square(optvar.com.eval_spline(self.T_c, "x", i_c-1) - optvar.com.eval_spline(         0, "x", i_c))
            J += np.square(optvar.com.eval_spline(         0, "x", i_c) - optvar.com.eval_spline(self.T_c/2, "x", i_c))
            J += np.square(optvar.com.eval_spline(self.T_c/2, "x", i_c) - optvar.com.eval_spline(  self.T_c, "x", i_c))
            J += np.square(optvar.com.eval_spline(self.T_c, "y", i_c-1) - optvar.com.eval_spline(         0, "y", i_c))
            J += np.square(optvar.com.eval_spline(         0, "y", i_c) - optvar.com.eval_spline(self.T_c/2, "y", i_c))
            J += np.square(optvar.com.eval_spline(self.T_c/2, "y", i_c) - optvar.com.eval_spline(  self.T_c, "y", i_c))
        '''
        # experimental: penalize high accelerations of COM
        for i_c in range(0, self.n_c):
            J += np.square(optvar.com.eval_spline(         0, "x", i_c))
            J += np.square(optvar.com.eval_spline(  self.T_c, "x", i_c))
            J += np.square(optvar.com.eval_spline(self.T_c/2, "x", i_c))
            J += np.square(optvar.com.eval_spline(         0, "y", i_c))
            J += np.square(optvar.com.eval_spline(  self.T_c, "y", i_c))
            J += np.square(optvar.com.eval_spline(self.T_c/2, "y", i_c))
            
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

    
    
    