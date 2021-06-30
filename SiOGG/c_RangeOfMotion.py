# -*- coding: utf-8 -*-
"""

Ensure feasible kinematic motion by keeping the foot tips in range of motion

Area where foot tip can be is approximated by a rectangle around the foot's nominal position:

            -r < p_f[t] - c[t] - p_{nom} < r            (eq. 15)      
            
   <=>          |p_f[t] - c[t] - p_{nom}| < r
   <=>                           distance < r
   
   <=>      g(w, t) = /   distance - r      for distance > r
                      \   0                 otherwise
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
        self.r = problem.r
        self.n = problem.n
        self.n_c = problem.n_s
        self.n_s = problem.n_s
        self.n_f = problem.n_f
        self.T_c = problem.T_c
        self.n_optvar = problem.n_optvar
        self.problem = problem
        
        self.com = CenterOfMass(problem)
        self.p_nom_x = problem.p_nom[0]
        self.p_nom_y = problem.p_nom[1]
        
        
        
    def constraint(self, w):
        '''
        TODO
        constraint is checked in middle of each COM splie
        TODO: ceck if this is a good idea
        '''
        d = np.zeros((2*self.n_s*self.n_c))
        
        N = self.n_c*self.n_s
        for i_s in range(self.n_s):
            for i_c in range(self.n_c):
                idx = i_s*self.n_c + i_c
                d[  idx] = self.distance_feet_ROM(w, i_s, i_c, "x", self.p_nom_x)
                d[N+idx] = self.distance_feet_ROM(w, i_s, i_c, "y", self.p_nom_y)
                
        return d
    
    
    def distance_feet_ROM(self, w, k_s, k_c, dim, p_nom):
        '''
        return distance from foottips to rectangle around nominal foot position
        
        Parameters
        ----------
        w_p : OptvarFootPos, access to current foot locations
        c   : com position in respective dimension
        p_nom : (1 x n_f) np.ndarray, feet position in respective dimension

        Returns
        -------
        sum of distances
        '''
        w_p = OptvarFootPos(w, self.problem)
        feet_pos = w_p.get_feet_pos(k_s, dim)
        c = self.com.eval_spline(w, self.T_c/2, dim, k_c)
        
        distance = np.abs(feet_pos - c - p_nom)
        d = distance - self.r
        d[distance < self.r] = 0
        return np.sum(d)
        
        
    def grad_distance_feet_ROM(self, w, k_s, k_c, dim, p_nom):
        '''
        return gradient of dynamic equation
        '''
        w_p = OptvarFootPos(w, self.problem)
        eps = np.sqrt(np.finfo(float).eps)
        g = optimize.approx_fprime(w, self.distance_feet_ROM, eps, k_s, k_c, dim, p_nom)
        return g
    
    
    def jacobian(self, w):
        '''return jacobian of constraints'''
        jac = np.zeros((2*self.n_s*self.n_c, self.n_optvar))
        
        w_p = OptvarFootPos(w, self.problem)
        N = self.n_c*self.n_s
        
        for i_s in range(self.n_s):
            for i_c in range(self.n_c):
                idx = i_s*self.n_c + i_c
                jac[  idx] = self.grad_distance_feet_ROM(w, i_c, i_s, "x", self.p_nom_x)
                jac[N+idx] = self.grad_distance_feet_ROM(w, i_c, i_s, "y", self.p_nom_y)
                
        return jac
    
    
    def amount(self):
        '''
        return amount of constraint variables
        '''
        return self.n_s*self.n_c