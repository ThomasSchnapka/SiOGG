# -*- coding: utf-8 -*-
"""

Ensure that the optimized motion fullfills the dynamic equation (11)
making use of Simpsons law:
    
    dd_c[t] = f(x[t], u[t])
            = (g/h)*(c[t]-u[t])      for t € {t_k, (t_k+1-t_k)/2, t_k+1}
            
"""

import numpy as np
from CenterOfMass import CenterOfMass
from CenterOfPressure import CenterOfPressure

from scipy import optimize

class c_SystemDynamics:
    '''
    Constraint regarding initial final COM position
    '''
    def __init__(self, problem):
        self.n = problem.n
        self.n_s = problem.n_s
        self.n_w_c = problem.n_w_c
        self.n_w_p = problem.n_w_p
        self.n_w_u = problem.n_w_u
        self.x_com_0 = problem.x_com_0
        self.x_com_T = problem.x_com_T
        self.n_optvar = problem.n_optvar
        self.T = problem.T
        self.T_c = problem.T_c
        self.h = problem.h
        
        self.com = CenterOfMass(problem)
        self.cop = CenterOfPressure(problem)
        
        self.g = 9.81
        
        
    def constraint(self, w):
        '''
        TODO
        '''
        d = np.zeros(6*self.n*self.n_s)
        for i_c in range(self.n*self.n_s):
            d[3*i_c+0] = self.dynamic_equation(w,          0, "x", i_c)
            d[3*i_c+1] = self.dynamic_equation(w, self.T_c/2, "x", i_c)
            d[3*i_c+2] = self.dynamic_equation(w,   self.T_c, "x", i_c)
            d[3*i_c+3] = self.dynamic_equation(w,          0, "y", i_c)
            d[3*i_c+4] = self.dynamic_equation(w, self.T_c/2, "y", i_c)
            d[3*i_c+5] = self.dynamic_equation(w,   self.T_c, "y", i_c)
        return d
    
    
    def dynamic_equation(self, w, t_k, dim, k):
        '''
        return result of dynamic equation
        '''
        d = self.com.eval_spline(w, t_k, dim, k, der=2)
        d -= (self.g/self.h)*(  self.com.eval_spline(w, t_k, dim, k, der=0)
                              - self.cop.get_u(w, t_k, dim, k))
        return d
    
    
    def grad_dynamic_equation(self, w, t_k, dim, k):
        '''
        return gradient of dynamic equation
        '''
        eps = np.sqrt(np.finfo(float).eps)
        d = optimize.approx_fprime(w, self.dynamic_equation, eps, t_k, dim, k)
        
        #d = self.com.grad_eval_spline(t_k, dim, k, der=2)
        #d -= (self.g/self.h)*(  self.com.grad_eval_spline(t_k, dim, k, der=0)
        #                      - self.cop.grad_get_u(w, t_k, dim, k))
        return d
    
    
    def jacobian(self, w):
        '''return jacobian of constraints'''
        jac = np.zeros((6*self.n*self.n_s, self.n_optvar))
        
        for i_c in range(self.n*self.n_s):
            jac[3*i_c+0] = self.grad_dynamic_equation(w,          0, "x", i_c)
            jac[3*i_c+1] = self.grad_dynamic_equation(w, self.T_c/2, "x", i_c)
            jac[3*i_c+2] = self.grad_dynamic_equation(w,   self.T_c, "x", i_c)
            jac[3*i_c+3] = self.grad_dynamic_equation(w,          0, "y", i_c)
            jac[3*i_c+4] = self.grad_dynamic_equation(w, self.T_c/2, "y", i_c)
            jac[3*i_c+5] = self.grad_dynamic_equation(w,   self.T_c, "y", i_c)
        return jac
    
    
    def amount(self):
        '''
        return amount of constraint variables
        '''
        return 6*self.n*self.n_s