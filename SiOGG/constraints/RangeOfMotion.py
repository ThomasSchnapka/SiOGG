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

from scipy import optimize

class RangeOfMotion:
    '''
    Constraint regarding range of motion of the foot tips
    '''
    def __init__(self, problem):
        self.r = problem.r
        self.n_c = problem.n_c
        self.n_s = problem.n_s
        self.n_f = problem.n_f
        self.T_c = problem.T_c
        self.n_optvar = problem.n_optvar
        self.problem = problem
        
        #self.com = CenterOfMass(problem)
        self.p_nom_x = problem.p_nom[0]
        self.p_nom_y = problem.p_nom[1]
        
        
        
    def constraint(self, optvar):
        '''
        TODO
        constraint is checked in middle of each COM spline
        TODO: ceck if this is a good idea
        '''
        N = self.n_c*self.n_s*self.n_f
        d = np.zeros((2*N))
        
        
        for i_s in range(self.n_s):
            foot_pos_x = optvar.footpos.get_foot_pos(i_s, "x")
            foot_pos_y = optvar.footpos.get_foot_pos(i_s, "y")
            for i_c in range(self.n_c):
                idx = (i_s*self.n_c + i_c)*self.n_f
                c_x = optvar.com.eval_spline(self.T_c/2, "x", i_c)
                c_y = optvar.com.eval_spline(self.T_c/2, "y", i_c)
                for i_f in range(self.n_f):
                    d[  idx+i_f] = self.distance_feet_ROM(foot_pos_x[i_f], c_x, self.p_nom_x[i_f])
                    d[N+idx+i_f] = self.distance_feet_ROM(foot_pos_y[i_f], c_y, self.p_nom_y[i_f])
        return d
    
    
    def distance_feet_ROM(self, foot_pos, c, p_nom):
        '''
        return distance from foottips to rectangle around nominal foot position
        
        Parameters
        ----------
        foot_pos : position of foot
        p_nom : nominal foot position of evaluated foot in evaluated dimension
        
        Returns
        -------
        distance to nominal foot position
        '''
        
        
        ## linear
        #d = np.abs(foot_pos - c - p_nom) - self.r
        #if d < 0:
        #    d = 0
        
        ## quadtratic
        d = 0
        if ((c+p_nom) - foot_pos) > self.r:
            d = (foot_pos - (c+p_nom-self.r))**2
        elif (foot_pos - (c+p_nom)) > self.r:
            d = (foot_pos - (c+p_nom+self.r))**2
        return d
        
    def conv_grad_constraint(self, w, foot_pos, c, p_nom):
        '''converts w to OptVar in order to be used by approx_fprime'''
        optvar = OptVar(w, self.problem)
        d = self.constraint(optvar)
        return d
        
    
    def grad_constraint(self, w, foot_pos, c, p_nom):
        '''
        return gradient of dynamic equation
        '''
        eps = np.sqrt(np.finfo(float).eps)
        g = optimize.approx_fprime(w, self.conv_grad_distance_feet_ROM, eps, k_s, k_c, k_f, dim, p_nom)
        return g
    
    
    def jacobian(self, w):
        '''return jacobian of constraints'''
        N = self.n_c*self.n_s*self.n_f
        jac = np.zeros((N, self.n_optvar))
        
        for i_s in range(self.n_s):
            foot_pos_x = optvar.footpos.get_foot_pos(i_s, "x")
            foot_pos_y = optvar.footpos.get_foot_pos(i_s, "y")
            for i_c in range(self.n_c):
                idx = (i_s*self.n_c + i_c)*self.n_f
                c_x = optvar.com.eval_spline(self.T_c/2, "x", i_c)
                c_y = optvar.com.eval_spline(self.T_c/2, "y", i_c)
                for i_f in range(self.n_f):
                    d[  idx+i_f] = self.grad_constraint(w, foot_pos_x[i_f], c_x, self.p_nom_x[i_f])
                    d[N+idx+i_f] = self.grad_constraint(w, foot_pos_y[i_f], c_y, self.p_nom_y[i_f])
        return jac
    
    
    def amount(self):
        '''
        return amount of constraint variables
        '''
        return 2*self.n_s*self.n_c*self.n_f