# -*- coding: utf-8 -*-
"""

Ensure feasible kinematic motion by keeping the foot tips in range of motion

Area where foot tip can be is approximated by a rectangle around the foot's nominal position:

         -r < p_f[t] - c[t] - p_{nom} < r            (eq. 15)        
"""

import numpy as np
from CenterOfMass import CenterOfMass
from OptvarFootPos import OptvarFootPos

from scipy import optimize

class c_RangeOfMotion:
    '''
    Constraint regarding range of motion of the foot tips
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
        
        
        
    def constraint(self, w):
        '''
        TODO
        constraint is checked in middle of each COM splie
        TODO: ceck if this is a good idea
        '''
        w_p = OptvarFootPos(w, self.problem)
        
        d = np.zeros((2*self.n_s*self.n_c, self.n_f))
        
        for i_s in range(self.n_s):
            for i_c in range(self.n_c):
                d[i_s*self.n_c + i_c :] = distance_feet_ROM(w_p, i_s, i_c, "x")
                d[i_s*self.n_c + i_c + self.n_c*self.n_s, :] = distance_feet_ROM(w_p, i_s, i_c, "y")
                
        return np.ravel(d)
    
    
    def distance_feet_ROM(self, w_p, k_s, k_c, dim):
        '''
        return distance from foottips to rectangle around nominal foot position
        
        
        Parameters
        ----------
        w   : OptvarFootPos, access to current foot locations
        k_s : int, number of step
        k_c : int, number of com spline (relative to number of step)
        dim : dimension to evaluate, either "x" oder "y"

        Returns
        -------
        vector with distances for all n_f feet

        '''
        idx_dim = 0 if dim=="x" else 1
        feet_pos = w_p.get_feet_pos(i_s, dim)
        distance = (  feet_pos
                    - com.eval_spline(w, self.T_c/2, dim, i_s*i_c)
                    - self.p_nom[idx_dim])
        distance[np.abs(distance) < r] = 0
        return distance
        
        
    def grad_distance_feet_ROM(self, w_p, k_s, k_c, dim):
        '''
        return gradient of dynamic equation
        '''
        w_p = OptvarFootPos(w, self.problem)
        
        eps = np.sqrt(np.finfo(float).eps)
        d = optimize.approx_fprime(w, self.distance_feet_ROM, eps, w_p, k_s, k_c, dim)
        
        return d
    
    
    def jacobian(self, w):
        '''return jacobian of constraints'''
        jac = np.zeros((6*self.n*self.n_s, self.n_optvar))
        
        #for i_c in range(self.n*self.n_s):
            #jac[6*i_c+0] = self.grad_dynamic_equation(w,          0, "x", i_c)
            #jac[6*i_c+1] = self.grad_dynamic_equation(w, self.T_c/2, "x", i_c)
            #jac[6*i_c+2] = self.grad_dynamic_equation(w,   self.T_c, "x", i_c)
            #jac[6*i_c+3] = self.grad_dynamic_equation(w,          0, "y", i_c)
            #jac[6*i_c+4] = self.grad_dynamic_equation(w, self.T_c/2, "y", i_c)
            #jac[6*i_c+5] = self.grad_dynamic_equation(w,   self.T_c, "y", i_c)
            
        # WEITER MIT GRADIENT
        
        for i_s in range(self.n_s):
            for i_c in range(self.n_c):
                d[i_s*self.n_c + i_c, :] = distance_feet_ROM(w_p, i_s, i_c, "x")
                d[i_s*self.n_c + i_c + self.n_c*self.n_s, :] = distance_feet_ROM(w_p, i_s, i_c, "y")
                
                
        return jac
    
    
    def amount(self):
        '''
        return amount of constraint variables
        '''
        return self.n_s*self.n_c*self.n_f