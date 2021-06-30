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
from OptvarVertexWeight import OptvarVertexWeight
from OptvarFootPos import OptvarFootPos

class CenterOfPressure:
    def __init__(self, problem):
        self.problem = problem
        self.n = problem.n          # number of COM splines per step
        self.n_s = problem.n_s      # number of steps
        self.n_f = problem.n_f      # number of feet
        self.n_u = problem.n_u      # number of lambda (vertex weights) PER STEP
        self.n_w_c = problem.n_w_c  # number of com optimization variables
        self.n_w_p = problem.n_w_p  # number of total leg position variables
        self.n_w_u = problem.n_w_u  # number of total lambda values
        self.n_optvar = problem.n_optvar
        
        self.T_c = problem.T_c      # time spend in each COM spline
       
    # DEPRECATED
    #def get_u(self, w, dim):
    #    '''return control vector in desired dimension "x" or "y"'''
    #    assert(dim=="x" or dim=="y")
    #    w_u = OptvarControl(w, self.problem)
    #    w_p = OptvarLegPos(w, self.problem)
    #    u = np.zeros(self.n_u*self.n_s)
    #    
    #    # iterate over all steps of constant lambda intervals
    #    for i_s in range(self.n_s):
    #        w_p_s = w_p.get_feet_pos(i_s, dim)
    #        for i_u in range(self.n_u):
    #            lambda_u = w_u.get_lambda_u(i_s, i_u)
    #            u[i_s*self.n_u + i_u] = self.calculate_u(w_p_s, lambda_u)
    #    return u
    
    #def calculate_u(self, w_p_s, lambda_u):
    #    '''calculate u for a single step of constant lambda(t) (eq. 17)'''
    #    u = 0
    #    for i_f in range(self.n_f):
    #        u += lambda_u@w_p_s
    #    return u
    
    def get_u(self, w, t_k, dim, k):
        '''
        Return value of COP in desired spline interval and dimension

        Parameters
        ----------
        w : whole optimization vector
        t_k : local spline time
        dim : dimension to evaluate, either "x" oder "y"
        k : number of spline segment to evaluate

        Returns
        -------
        val : desired value of spline segment

        '''
        assert(dim=="x" or dim=="y")
        
        w_u = OptvarVertexWeight(w, self.problem)
        w_p = OptvarFootPos(w, self.problem)
        
        # determine which u segment is corresponding to t_k
        #k_s = int(0.999*k/self.n)                         # number of step
        #k_u = self.n_u*k_s + int(self.n_u*0.999*t_k/self.T_c)  # number of lambda vector
        k_s = int(k//self.n)                         # number of step
        k_u = self.n_u*k_s + int((self.n_u**t_k)//self.T_c)  # number of lambda vector

        feet_pos = w_p.get_feet_pos(k_s, dim)
        lambda_u = w_u.get_lambda_u(k_u)
        
        val = lambda_u@feet_pos 
        
        return val
    

    def grad_get_u(self, w, t_k, dim, k):
        '''
        return gradient of COP under the given parameters
        
        get_u depends on leg positions and vertex weights, which are both
        present in optimization variable. Thus, instead of setting only the
        current variable to one and all remaining to zero, we apply this
        technique blockwise. E.g. calculating the gradient wrt to w_u, we set
        the current element in w_u to 1, the remaining elements in w_u to
        0 and let all other variables stay the same.
        
        k is from 0 to n-1
        '''
        # only calculate gradiend wrt w_c as all other elements are zero
        grad = np.zeros(self.n_optvar)
        
        # part 1: gradient wrt w_p
        w_iter = np.copy(w)
        w_iter[self.n_w_c:self.n_w_u] = 0
        for i in range(self.n_w_c, self.n_w_c+self.n_w_p):
            w_iter[i] = 1
            grad[i] = self.get_u(w_iter, t_k, dim, k)
            w_iter[i] = 0
            
        # part 2: gradient wrt w_u
        w_iter = np.copy(w)
        w_iter[self.n_w_u:] = 0
        for i in range(self.n_w_c+self.n_w_p, self.n_optvar):
            w_iter[i] = 1
            grad[i] = self.get_u(w_iter, t_k, dim, k)
            w_iter[i] = 0
        return grad
    
        
        


