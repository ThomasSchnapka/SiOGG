# -*- coding: utf-8 -*-
"""
Per sampling point the sum of all vertex weights must equal 1
"""

import numpy as np

class UnilateralForces:
    '''
    Constraint regarding unilateral forces
    '''
    def __init__(self, problem, tol=1e-4):
        self.n_s = problem.n_s
        self.n_c = problem.n_c
        self.n_f = problem.n_f
        self.n_w_non_u = problem.n_w_c + problem.n_w_p
        self.n_optvar = problem.n_optvar
        
        self.tol = tol              # tolerance for upper/lower bounds
        
        
    def constraint(self, optvar):
        '''
        TODO
        '''
        d = np.zeros(self.n_s*self.n_c*3)
        for i_c in range(self.n_c*self.n_s):
            for i_u in range(3):
                d[i_c*3 + i_u] = np.sum(optvar.vertexweight.get_lambda(i_c, i_u))-1.0
        return d
    
    def jacobian(self, optvar):
        jac = np.zeros((self.n_s*self.n_c*3, self.n_optvar))
        for i_c in range(self.n_c*self.n_s):
           for i_u in range(3):
               idx = i_c*3 + i_u
               jac[self.n_w_non_u + idx*self.n_f:self.n_w_non_u + (idx+1)*self.n_f] = 1
        return jac
        
    
    def amount(self):
        '''return amount of constraint variables'''
        return self.n_s*self.n_c*3
    
    def constraint_bound_lower(self):
        '''return lower constraint bound'''
        return np.ones(self.amount())*(-1)*self.tol
    
    def constraint_bound_upper(self):
        '''return upper constraint bound'''
        return np.ones(self.amount())*self.tol
    