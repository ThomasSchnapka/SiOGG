# -*- coding: utf-8 -*-
"""
construction of optimization variable w_u:
    w_u = [w_u_1, ..., w_u_{n_s}]
    w_u_s = [lambda_1, ..., lambda_{n_u}]
    lambda_u = [lambda_u^1, ..., lambda_u^{n_f}]
"""

import numpy as np

class VertexWeight:
    '''class for easier access to control optimization variable w_u'''
    def __init__(self, problem, w_u):
        self.problem  =problem
        self.n_s = problem.n_s     # number of steps
        self.n_f = problem.n_f     # number of feet
        self.n_w_u = problem.n_w_u  # number of total lambda values
        
        self.w_u = w_u
        assert(len(w_u) == problem.n_w_u)
        
    def get_lambda(self, k_c, k_u):
        '''
        return lambda value number k_u in com spline k_c for all n_f feet

        Parameters
        ----------
        k_c : number of COM spline
        k_u : number of lambda vector inside COM spline (k_u <= 3)
        
        Returns
        ----------
        lambda : (1 x n_f) vector of weights for each foot
        
        '''
        assert(k_c < self.problem.n_s*self.problem.n_c)
        assert(k_u <= 3)
        return self.w_u[self.n_f*k_u:self.n_f*(k_u+1)]
    
    
    def grad_get_lambda(self, k_c, k_u):
        '''
        gradient of get_lambda() wrt w_u
        
        '''
        assert(k_c < self.problem.n_s*self.problem.n_c)
        assert(k_u <= 3)
        grad = np.zeros(self.n_w_u)
        idx = np.zeros((self.n_f), dtype=int)
        for i_f in range(self.n_f):
            idx[i_f] = self.n_f*k_u + i_f
            grad[idx[i_f]] = 1
        return grad, idx