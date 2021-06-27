# -*- coding: utf-8 -*-
"""
construction of optimization variable w_u:
    w_u = [w_u_1, ..., w_u_{n_s}]
    w_u_s = [lambda_1, ..., lambda_{n_u}]
    lambda_u = [lambda_u^1, ..., lambda_u^{n_f}]
"""

import numpy as np

class OptvarControl:
    '''class for easier access to control optimization variable w_u'''
    def __init__(self, w, problem):
        self.problem  =problem
        self.n_s = problem.n_s     # number of steps
        self.n_f = problem.n_f     # number of feet
        self.n_u = problem.n_u     # number of lambda (vertex weights) PER STEP
        self.n_w_u = problem.n_w_u  # number of total lambda values
        
        self.w_u = w[-self.n_w_u:]
      
    # DEPRECATED
    #def get_lambda_u(self, k_s, k_u):
    #    '''
    #    return lambda values for a single interval of constant lambda for all
    #    n_f feet
    #
    #    Parameters
    #    ----------
    #    k_s : number of step
    #    k_u : (local) number of u
    #    
    #    Returns
    #    ----------
    #    lambda : (1 x n_f) vector of weights for each foot
    #    
    #    '''
    #    return self.w_u[self.n_f*(self.n_u*k_s + k_u):self.n_f*(self.n_u*k_s + k_u + 1)]
        
    def get_lambda_u(self, k_u):
        '''
        return lambda values for a single interval of constant lambda for all
        n_f feet

        Parameters
        ----------
        k_u : (global) number of lambda vector
        
        Returns
        ----------
        lambda : (1 x n_f) vector of weights for each foot
        
        '''
        return self.w_u[self.n_f*k_u:self.n_f*(k_u+1)]