# -*- coding: utf-8 -*-
"""
Constraint regarding initial final COM position
"""

import numpy as np

from CenterOfMass import CenterOfMass

class c_COMIniFiPos:
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
        self.n = problem.n
        self.T = problem.T
        self.T_c = problem.T_c
        
        self.com = CenterOfMass(problem)
    
    def constraint(self, w):
        '''return constraints'''
        #w_c = w[:self.n_w_c]
        #c_inifi_com = w_c[[0, 1, 2, 3, -10, -9, -8, -7]] - np.hstack((self.x_com_0, self.x_com_T))
        const = np.zeros(8)
        const[0] = self.com.get_c(w,      0, "x", 0) - self.x_com_0[0, 0]
        const[1] = self.com.get_c(w,      0, "x", 1) - self.x_com_0[0, 1]
        const[2] = self.com.get_c(w,      0, "y", 0) - self.x_com_0[1, 0]
        const[3] = self.com.get_c(w,      0, "y", 1) - self.x_com_0[1, 1]
        const[4] = self.com.get_c(w, self.T, "x", 0) - self.x_com_T[0, 0]
        const[5] = self.com.get_c(w, self.T, "x", 1) - self.x_com_T[0, 1]
        const[6] = self.com.get_c(w, self.T, "y", 0) - self.x_com_T[1, 0]
        const[7] = self.com.get_c(w, self.T, "y", 1) - self.x_com_T[1, 1]
        return const
    
    def jacobian(self, w):
        '''return jacobian of constraints'''
        #jac = np.zeros((8, self.n_optvar))
        #jac[0,0] = 1
        #
        jac = np.zeros((8, self.n_optvar))
        jac[0] = self.com.grad_get_c(     0, "x", 0)
        jac[1] = self.com.grad_get_c(     0, "x", 1)
        jac[2] = self.com.grad_get_c(     0, "y", 0)
        jac[3] = self.com.grad_get_c(     0, "y", 1)
        jac[4] = self.com.grad_get_c(self.T, "x", 0)
        jac[5] = self.com.grad_get_c(self.T, "x", 1)
        jac[6] = self.com.grad_get_c(self.T, "y", 0)
        jac[7] = self.com.grad_get_c(self.T, "y", 1)
        return jac
    
    def amount(self):
        '''return amount of constraint variables'''
        return 8