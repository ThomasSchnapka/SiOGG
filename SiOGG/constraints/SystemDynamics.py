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
    
    
    def fill_jacobian(self, jac, optvar):
        '''jacobian of constraint'''
        jac.prepare(self.amount())
        
        for i_c in range(self.n_c*self.n_s):
            for i_u in range(3):
                jac.fill('c', 6*i_c+i_u  , self.grad_c_dynamic_equation(optvar, i_c, i_u, "x"))
                jac.fill('p', 6*i_c+i_u  , self.grad_p_dynamic_equation(optvar, i_c, i_u, "x"))
                jac.fill('u', 6*i_c+i_u  , self.grad_u_dynamic_equation(optvar, i_c, i_u, "x"))
                jac.fill('c', 6*i_c+i_u+3, self.grad_c_dynamic_equation(optvar, i_c, i_u, "y"))
                jac.fill('p', 6*i_c+i_u+3, self.grad_p_dynamic_equation(optvar, i_c, i_u, "y")) 
                jac.fill('u', 6*i_c+i_u+3, self.grad_u_dynamic_equation(optvar, i_c, i_u, "y"))
        jac.release()
        
    
    def dynamic_equation(self, optvar, k_c, k_u, dim):
        '''
        return result of dynamic equation
        k_u : number of COP inside COM spline (for each k_c there are 3 k_u)
        '''
        assert(k_u < 3)
        t_k = k_u*self.T_c*0.5 # local spline time
        d = optvar.com.eval_spline(t_k, dim, k_c, der=2)
        d -= (self.g/self.h)*(  optvar.com.eval_spline(t_k, dim, k_c, der=0)
                              - optvar.cop.get_cop(k_c, k_u, dim))
        assert(type(d)==np.float64)
        return d
    
    
    def grad_c_dynamic_equation(self, optvar, k_c, k_u, dim):
        '''
        gradient of dynamic equation wrt COM position
        '''
        assert(k_u < 3)
        t_k = k_u*self.T_c*0.5
        grad = optvar.com.grad_eval_spline(t_k, dim, k_c, der=2)
        grad -= (self.g/self.h)*(optvar.com.grad_eval_spline(t_k, dim, k_c, der=0))
        return grad
    
    
    def grad_p_dynamic_equation(self, optvar, k_c, k_u, dim):
        '''
        gradient of dynamic equation wrt w_p
        '''
        assert(k_u < 3)
        t_k = k_u*self.T_c*0.5
        grad = -(self.g/self.h)*optvar.cop.grad_p_get_cop(k_c, k_u, dim)
        return grad
    
    
    def grad_u_dynamic_equation(self, optvar, k_c, k_u, dim):
        '''
        gradient of dynamic equation wrt w_u
        '''
        assert(k_u < 3)
        t_k = k_u*self.T_c*0.5
        grad = -(self.g/self.h)*optvar.cop.grad_u_get_cop(k_c, k_u, dim)
        return grad
    
    
    def amount(self):
        '''
        return amount of constraint variables
        '''
        return 6*self.n_c*self.n_s