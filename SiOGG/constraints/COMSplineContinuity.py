# -*- coding: utf-8 -*-
"""
Constraint regarding continuity of COM spline
Ensure that spline knots overlap and that the COM velocity at each junction is 
equal
"""

import numpy as np


class COMSplineContinuity:
    '''
    Constraint regarding continuity of COM spline
    '''
    def __init__(self, problem):
        #self.n_w_c = problem.n_w_c
        #self.n_w_p = problem.n_w_p
        #self.n_w_u = problem.n_w_u
        self.T_c = problem.T_c
        self.n_junct = problem.n_c*problem.n_s-1
        self.n_optvar = problem.n_optvar
        
        
    
    def constraint(self, optvar):
        '''return distance between knots in order 
        [[x], [y], [dx/dt], [dy/dt]]
        with [x] = [c_x_1(T) - c_x_2(0), ...]
        '''
        d = np.zeros(4*self.n_junct)
        for i in range(self.n_junct):
            d[0*self.n_junct+i] = (
                       optvar.com.eval_spline(self.T_c, "x",     i, 0)
                      -optvar.com.eval_spline(       0, "x", (i+1), 0)
                      )
            d[1*self.n_junct+i] = (
                       optvar.com.eval_spline(self.T_c, "y",     i, 0)
                      -optvar.com.eval_spline(       0, "y", (i+1), 0)
                      )
            d[2*self.n_junct+i] = (
                       optvar.com.eval_spline(self.T_c, "x",     i, 1)
                      -optvar.com.eval_spline(       0, "x", (i+1), 1)
                      )
            d[3*self.n_junct+i] = (
                       optvar.com.eval_spline(self.T_c, "y",     i, 1)
                      -optvar.com.eval_spline(       0, "y", (i+1), 1)
                      )
        return d
    
    
    def jacobian(self, optvar):
        '''return jacobian of constraints'''
        jac = np.zeros((4*self.n_junct, self.n_optvar))
        for i in range(self.n_junct):
            jac[0*self.n_junct+i] = (
                       optvar.com.grad_eval_spline(self.T_c, "x",     i, 0)
                      -optvar.com.grad_eval_spline(     0.0, "x", (i+1), 0)
                      )
            jac[1*self.n_junct+i] = (
                       optvar.com.grad_eval_spline(self.T_c, "y",     i, 0)
                      -optvar.com.grad_eval_spline(     0.0, "y", (i+1), 0)
                      )
            jac[2*self.n_junct+i] = (
                       optvar.com.grad_eval_spline(self.T_c, "x",     i, 1)
                      -optvar.com.grad_eval_spline(     0.0, "x", (i+1), 1)
                      )
            jac[3*self.n_junct+i] = (
                       optvar.com.grad_eval_spline(self.T_c, "y",     i, 1)
                      -optvar.com.grad_eval_spline(     0.0, "y", (i+1), 1)
                      )
        return jac
    
    
    def fill_jacobian(self, jac, optvar):
        '''return jacobian of constraints'''
        jac.prepare(self.amount())
        for i in range(self.n_junct):
            jac.fill('c', 0*self.n_junct+i, (
                       optvar.com.grad_eval_spline(self.T_c, "x",     i, 0)
                      -optvar.com.grad_eval_spline(     0.0, "x", (i+1), 0)
                      ))
            jac.fill('c', 1*self.n_junct+i, (
                       optvar.com.grad_eval_spline(self.T_c, "y",     i, 0)
                      -optvar.com.grad_eval_spline(     0.0, "y", (i+1), 0)
                      ))
            jac.fill('c', 2*self.n_junct+i, (
                       optvar.com.grad_eval_spline(self.T_c, "x",     i, 1)
                      -optvar.com.grad_eval_spline(     0.0, "x", (i+1), 1)
                      ))
            jac.fill('c', 3*self.n_junct+i, (
                       optvar.com.grad_eval_spline(self.T_c, "y",     i, 1)
                      -optvar.com.grad_eval_spline(     0.0, "y", (i+1), 1)
                      ))
    
    
    def amount(self):
        '''
        return amount of constraint variables
        here: number of junctions of COM splines
        '''
        return int(2*2*self.n_junct)
        


