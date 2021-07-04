# -*- coding: utf-8 -*-
"""

Ensure that the optimized motion fullfills the dynamic equation (11)
making use of Simpsons law:
    
    dd_c[t] = f(x[t], u[t])
            = (g/h)*(c[t]-u[t])      for t € {t_k, (t_k+1-t_k)/2, t_k+1}
            
"""

import numpy as np

from scipy import optimize

class SystemDynamics:
    '''
    Constraint regarding initial final COM position
    '''
    def __init__(self, problem):
        self.n_c = problem.n_c
        self.n_s = problem.n_s
        #self.n_w_c = problem.n_w_c
        #self.n_w_p = problem.n_w_p
        #self.n_w_u = problem.n_w_u
        #self.x_com_0 = problem.x_com_0
        #self.x_com_T = problem.x_com_T
        self.n_optvar = problem.n_optvar
        self.T = problem.T
        self.T_c = problem.T_c
        self.h = problem.h
        
        
        self.g = 9.81
        
        
    def constraint(self, optvar):
        '''
        TODO
        '''
        d = np.zeros(6*self.n_c*self.n_s)
        for i_c in range(self.n_c*self.n_s):
            for i_u in range(3):
                d[6*i_c+i_u  ] = self.dynamic_equation(optvar, i_c, i_u, "x")
                d[6*i_c+i_u+3] = self.dynamic_equation(optvar, i_c, i_u, "y")
        return d
    
    
    def dynamic_equation(self, optvar, k_c, k_u, dim):
        '''
        return result of dynamic equation
        k_u : number of COP inside COM spline (for each k_c there are 3 k_d)
        '''
        assert(k_u < 3)
        t_k = k_u*self.T_c/2
        d = 0.0+optvar.com.eval_spline(t_k, dim, k_c, der=2)
        d -= (self.g/self.h)*(  optvar.com.eval_spline(t_k, dim, k_c, der=0)
                              - optvar.cop.get_u(k_c, k_u, dim))
        assert(type(d)==np.float64)
        return d
    
    
    def grad_dynamic_equation(self, w, t_k, dim, k):
        '''
        return gradient of dynamic equation
        '''
        eps = np.sqrt(np.finfo(float).eps)
        d = optimize.approx_fprime(w, self.dynamic_equation, eps, t_k, dim, k)
        
        #d1 = self.com.grad_eval_spline(t_k, dim, k, der=2)
        #d1 -= (self.g/self.h)*(  self.com.grad_eval_spline(t_k, dim, k, der=0)
        #                       - self.cop.grad_get_u(w, t_k, dim, k))
        #print(d-d1)
        return d
    
    
    def jacobian(self, w):
        '''return jacobian of constraints'''
        jac = np.zeros((6*self.n_c*self.n_s, self.n_optvar))
        
        for i_c in range(self.n_c*self.n_s):
            jac[6*i_c+0] = self.grad_dynamic_equation(w,          0, "x", i_c)
            jac[6*i_c+1] = self.grad_dynamic_equation(w, self.T_c/2, "x", i_c)
            jac[6*i_c+2] = self.grad_dynamic_equation(w,   self.T_c, "x", i_c)
            jac[6*i_c+3] = self.grad_dynamic_equation(w,          0, "y", i_c)
            jac[6*i_c+4] = self.grad_dynamic_equation(w, self.T_c/2, "y", i_c)
            jac[6*i_c+5] = self.grad_dynamic_equation(w,   self.T_c, "y", i_c)
        return jac
    
    
    def amount(self):
        '''
        return amount of constraint variables
        '''
        return 6*self.n_c*self.n_s