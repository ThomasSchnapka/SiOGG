# -*- coding: utf-8 -*-


import numpy as np

class CenterOfMass:
    '''
    Object holding the quartic COM spline
    '''
    def __init__(self, problem):
        self.n = problem.n  # number of COM polynomials per dimension per step
        self.n_s = problem.n_s # number of steps
        self.T_c = problem.T_c # time spend in spline
        self.n_w_c = problem.n_w_c # number of COM spline coefficients
        self.n_optvar = problem.n_optvar # number of optimization variables
        self.n_junct = problem.n-1 # number of junctions
        
        
    def get_c(self, w, t, dim, der=0):
        '''returns c(t) value at time t in dimension dim'''
        k = self.find_k(t)
        t_loc = t - k*self.T_c
        c = self.eval_spline(w, t_loc, dim, k, der)
        return c


    def find_k(self, t):
        '''
        return the number of the spline segment in action at time t
        
        rounding is neccessary to prevent decimal errors caused by the 
        base two representation
        '''
        k = None
        # testing a new approach
        '''
        for i in range(self.n*self.n_s):
            if (    (round(t, 3) >= round(self.T_c*i,     3)) 
                and (round(t, 3) <= round(self.T_c*(i+1), 3))):
                k = i
                break
        if k == None:
            raise ValueError(f"wrong k for t: {t}")
        '''
        k = int(0.999*t/self.T_c)
        assert(k<=self.n*self.n_s)
        return k
    
    
    def eval_spline(self, w, t_k, dim, k, der=0):
        '''
        Return value of desired spline segment k at local time t_c

        Parameters
        ----------
        w : whole optimization vector
        t_k : local spline time
        dim : dimension to evaluate, either "x" oder "y"
        k : number of spline segment to evaluate
        der : deritvative to calculate. Default is 0.

        Returns
        -------
        val : desired value of spline segment

        '''
        w_c = w[:self.n_w_c]
        if dim=="x":
            w_c = w_c[:int(self.n_w_c*0.5)]
        elif dim=="y":
            w_c = w_c[int(self.n_w_c*0.5):]
        else:
            raise ValueError(f"wrong dimension! + {dim}")
        a_k = w_c[(self.n*k):(self.n*k+5)]   
        if der==0:
            val = (  a_k[0] 
                   + a_k[1]*t_k 
                   + a_k[2]*t_k**2 
                   + a_k[3]*t_k**3 
                   + a_k[4]*t_k**4
                   )
        elif der==1:
            val = (    a_k[1] 
                   + 2*a_k[2]*t_k 
                   + 3*a_k[3]*t_k**2 
                   + 4*a_k[4]*t_k**3
                   ) 
        elif der==2:
            val = (  2*a_k[2] 
                   + 6*a_k[3]*t_k**1 
                   +10*a_k[4]*t_k**2
                   ) 
        else:
            raise ValueError(f"wrong derivative! + {dim}")    
        return val
    
    def grad_get_c(self, t, dim, der=0):
        '''
        return gradient of get_c() under the given parameters
        
        making use of the linear relationship between the optimization
        variable and the spline, each element of the gradient can be calculated
        by setting the corresponding optvar element to one and all others to
        zero and evaluationg the spline
        
        k is from 0 to n-1
        '''
        # only calculate gradiend wrt w_c as all other elements are zero
        grad = np.zeros(self.n_optvar)
        w_iter = np.zeros(self.n_optvar)
        for i in range(self.n_w_c):
            w_iter[i] = 1
            grad[i] = self.get_c(w_iter, t, dim, der)
            w_iter[i] = 0
        return grad
        
    
    def grad_eval_spline(self, t_k, dim, k, der=0):
        '''
        return gradient of eval_spline() under the given parameters
        
        making use of the linear relationship between the optimization
        variable and the spline, each element of the gradient can be calculated
        by setting the corresponding optvar element to one and all others to
        zero and evaluationg the spline
        
        k is from 0 to n-1
        '''
        # only calculate gradiend wrt w_c as all other elements are zero
        grad = np.zeros(self.n_optvar)
        w_iter = np.zeros(self.n_optvar)
        for i in range(self.n_w_c):
            w_iter[i] = 1
            grad[i] = self.eval_spline(w_iter, t_k, dim, k, der)
            w_iter[i] = 0
        return grad

                    
            
            
        
        
        
        
        