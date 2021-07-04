# -*- coding: utf-8 -*-
"""
Per sampling point the sum of all vertex weights must equal 1
"""

import numpy as np

class UnilateralForces:
    '''
    Constraint regarding unilateral forces
    '''
    def __init__(self, problem):
        self.n_s = problem.n_s
        self.n_c = problem.n_c
        self.n_f = problem.n_f
        self.n_w_non_u = problem.n_w_c + problem.n_w_p
        self.n_optvar = problem.n_optvar
        
        
    def constraint(self, optvar):
        '''
        TODO
        '''
        d = np.zeros(self.n_s*self.n_c*3)
        for i_c in range(self.n_c*self.n_s):
            for i_u in range(3):
                d[i_c*3 + i_u] = 1.0-np.sum(optvar.vertexweight.get_lambda(i_c, i_u))
        return d
    
    def jacobian(self, optvar):
        jac = np.zeros((self.n_s*self.n_c*3, self.n_optvar))
        for i_c in range(self.n_c*self.n_s):
           for i_u in range(3):
               idx = i_c*3 + i_u
               jac[self.n_w_non_u + idx*self.n_f:self.n_w_non_u + (idx+1)*self.n_f] = 1
        return jac
        
    
    
    def amount(self):
        return self.n_s*self.n_c*3
    