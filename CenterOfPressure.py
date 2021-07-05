# -*- coding: utf-8 -*-
"""
Return control (CoP) vector (eq. 17 in https://arxiv.org/pdf/1705.10313.pdf)

construction of optimization variable w_u:
    w_u = [w_u_1, ..., w_u_{n_s}]
    w_u_{n_s} = [lambda_1, ..., lambda_{n_u}]
    lambda_{n_u} = [lambda_u^1, ..., lambda_u^{n_f}]
    
construction of optimization variable w_p:
    w_p = [w_s_1, ..., w_s_{n_s}]
    w_p^s = [p^s_1, ..., p^s_{n_f}]
    p^s_f = [p^s_f_x, p^s_f_y]
"""

import numpy as np

class CenterOfPressure:
    def __init__(self, problem, footpos, vertexweight):
        self.problem = problem
        self.footpos = footpos
        self.vertexweight = vertexweight
        self.n_c = problem.n_c          # number of COM splines per step
        self.n_s = problem.n_s      # number of steps
        self.n_f = problem.n_f      # number of feet
        #self.n_u = problem.n_u      # number of lambda (vertex weights) PER STEP
        #self.n_w_c = problem.n_w_c  # number of com optimization variables
        #self.n_w_p = problem.n_w_p  # number of total leg position variables
        self.n_w_u = problem.n_w_u  # number of total lambda values
        self.n_optvar = problem.n_optvar
        
        self.T_c = problem.T_c      # time spend in each COM spline
           
        
    def get_cop(self, k_c, k_u, dim):
        '''
        Return value of COP in desired spline interval and dimension

        Parameters
        ----------
        k_c : number of COM spline to evaluate in
        k_u : number of vertex weight in corresponding COM spline
        dim : dimension to evaluate, either "x" or "y"

        Returns
        -------
        val : desired value of spline segment

        '''
        assert(dim=="x" or dim=="y")
        assert(k_c <= self.problem.n_s*self.problem.n_c)
        assert(k_u <= 3)
        
        k_s = int(k_c//self.n_c)                    # number of step
        feet_pos = self.footpos.get_foot_pos(k_s, dim)
        lambda_u = self.vertexweight.get_lambda(k_c, k_u)
        
        val = lambda_u@feet_pos 
        
        return val
    

    def grad_p_get_cop(self, k_c, k_u, dim):
        '''
        gradient of get_u() w.r.t. w_p
        return gradient of COP under the given parameters
        
        '''
        assert(dim=="x" or dim=="y")
        assert(k_c <= self.problem.n_s*self.problem.n_c)
        assert(k_u <= 3)
        
        k_s = int(k_c//self.n_c)   # number of step
        grad_feet_pos, idx = self.footpos.grad_get_foot_pos(k_s, dim)
        lambda_u = self.vertexweight.get_lambda(k_c, k_u)
        grad = np.copy(grad_feet_pos)
        for i_f in range(self.n_f):
            grad[idx[i_f]] *= lambda_u[i_f]
        return grad
    
    
    def grad_u_get_cop(self, k_c, k_u, dim):
        '''
        gradient of get_u() w.r.t. w_p
        return gradient of COP under the given parameters
        
        '''
        assert(dim=="x" or dim=="y")
        assert(k_c <= self.problem.n_s*self.problem.n_c)
        assert(k_u <= 3)
        
        k_s = int(k_c//self.n_c)   # number of step
        feet_pos = self.footpos.get_foot_pos(k_s, dim)
        grad_lambda_u, idx = self.vertexweight.grad_get_lambda(k_c, k_u)
        
        grad = np.copy(grad_lambda_u)
        for i_f in range(self.n_f):
            grad[idx[i_f]] *= feet_pos[i_f]
        return grad
    
        
        


