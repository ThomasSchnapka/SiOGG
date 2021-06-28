# -*- coding: utf-8 -*-

import numpy as np


class CostFunction:
    '''
    CostFunction
    '''
    def __init__(self, problem):
        self.c = problem.c
        self.n_u = problem.n_u
        self.n_f = problem.n_f
        self.n_w_u = problem.n_w_u
        self.n_optvar = problem.n_optvar
        
        
    def cost(self, w):
        '''return cost associated with optimization variable w'''
        w_u = w[-self.n_w_u:]
        w_u_opt = (  np.tile(np.ravel(self.c), self.n_u) 
                   / np.tile(np.sum(self.c, axis=1), self.n_f*self.n_u))
        J = (w_u - w_u_opt).T@(w_u - w_u_opt)
        return J
    
    
    def gradient(self, w):
        '''return gradient of cost associated with w'''
        w_u = w[-self.n_w_u:]
        w_u_opt = (  np.tile(np.ravel(self.c), self.n_u) 
                   / np.tile(np.sum(self.c, axis=1), self.n_f*self.n_u))
        # create empty gradient vector and fill it
        dw = np.zeros(self.n_optvar)
        dw[-self.n_w_u:] = 2*w_u - 2*w_u_opt
        #return np.zeros(self.n_optvar)
        #print(dw.shape, dw)
        return dw

    
    
    