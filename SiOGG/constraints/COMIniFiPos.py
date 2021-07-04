# -*- coding: utf-8 -*-
"""
Constraint regarding initial final COM position
"""

import numpy as np

class COMIniFiPos:
    '''
    Constraint regarding initial final COM position
    '''
    def __init__(self, problem):
        self.n_s = problem.n_s
        self.n_w_c = problem.n_w_c
        self.n_w_p = problem.n_w_p
        self.n_w_u = problem.n_w_u
        self.x_com_0 = problem.x_com_0
        self.x_com_T = problem.x_com_T
        self.n_optvar = problem.n_optvar
        self.n_c = problem.n_c
        self.T = problem.T
        self.T_c = problem.T_c
    
    
    def constraint(self, optvar):
        '''return constraints'''
        #w_c = w[:self.n_w_c]
        #c_inifi_com = w_c[[0, 1, 2, 3, -10, -9, -8, -7]] - np.hstack((self.x_com_0, self.x_com_T))
        const = np.zeros(8)
        const[0] = optvar.com.get_c(   0.0, "x", 0) - self.x_com_0[0, 0]
        const[1] = optvar.com.get_c(   0.0, "x", 1) - self.x_com_0[0, 1]
        const[2] = optvar.com.get_c(   0.0, "y", 0) - self.x_com_0[1, 0]
        const[3] = optvar.com.get_c(   0.0, "y", 1) - self.x_com_0[1, 1]
        const[4] = optvar.com.get_c(self.T, "x", 0) - self.x_com_T[0, 0]
        const[5] = optvar.com.get_c(self.T, "x", 1) - self.x_com_T[0, 1]
        const[6] = optvar.com.get_c(self.T, "y", 0) - self.x_com_T[1, 0]
        const[7] = optvar.com.get_c(self.T, "y", 1) - self.x_com_T[1, 1]
        return const
    
    
    def fill_jacobian(self, jac, optvar):
        '''jacobian of constraint'''
        jac.prepare(self.amount())
        
        jac.fill('c', 0, optvar.com.grad_get_c(   0.0, "x", 0))
        jac.fill('c', 1, optvar.com.grad_get_c(   0.0, "x", 1))
        jac.fill('c', 2, optvar.com.grad_get_c(   0.0, "y", 0))
        jac.fill('c', 3, optvar.com.grad_get_c(   0.0, "y", 1))
        jac.fill('c', 4, optvar.com.grad_get_c(self.T, "x", 0))
        jac.fill('c', 5, optvar.com.grad_get_c(self.T, "x", 1))
        jac.fill('c', 6, optvar.com.grad_get_c(self.T, "y", 0))
        jac.fill('c', 7, optvar.com.grad_get_c(self.T, "y", 1))
    
    
    def amount(self):
        '''return amount of constraint variables'''
        return 8