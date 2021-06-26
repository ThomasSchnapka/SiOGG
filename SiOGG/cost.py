# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 20:02:08 2021

@author: z003p2nh
"""

import numpy as np


class CostFunction:
    '''
    CostFunction
    '''
    def __init__(self, problem):
        self.c = problem.c
        self.n_u = problem.n_u
        self.n_w_u = problem.n_w_u
        self.N = problem.N
        
    def cost(self, w):
        '''return cost associated with optimization variable w'''
        w_u = w[-self.n_w_u:]
        w_u_opt = np.tile((   np.ravel(self.c), self.n_u) 
                            / np.tile(np.sum(self.c, axis=1), 4*self.n_u))
        J = (w_u - w_u_opt).T@(w_u - w_u_opt)
        return J
    
    def gradient(self, w):
        '''return gradient of cost associated with w'''
        w_u = w[-self.n_w_u:]
        w_u_opt = np.tile((   np.ravel(self.c), self.n_u) 
                            / np.tile(np.sum(self.c, axis=1), 4*self.n_u))
        dw = np.zeros(self.N)
        dw[-self.n_w_u:] = 2*w_u - 2*w_u_opt
        return dw
    
    
    