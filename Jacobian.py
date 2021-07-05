# -*- coding: utf-8 -*-

import numpy as np


class Jacobian:
    '''
    Structure for easier access to the Jacobian
    
    The row_pointer is used to point to the current row in order to prevent
    the necessity for slising in the application where the jacobian is used
    '''
    def __init__(self, problem):
        self.n_optvar = problem.n_optvar
        self.n_constr = problem.n_constr
        self.n_w_c = problem.n_w_c
        self.n_w_p = problem.n_w_p
        self.n_w_u = problem.n_w_u
        
        self.jac = np.zeros((self.n_constr, self.n_optvar))
        self.row_ptr = 0
        self.n_fill = 0
        self.filling_in_progress = False
        
    
    def prepare(self, n):
        '''
        Tell object how many rows get filled during the next fill process.

        Parameters
        ----------
        n : int, number of constraint function the jacobian is provided for
                 during the next fill procedure

        Returns
        -------
        None.

        '''
        if self.filling_in_progress == True:
            raise RuntimeError("Jacobian was not released from last fill")
        self.filling_in_progress = True
        self.n_fill = n
        
        
    def release(self):
        '''
        Tell jacobian that current fill is completed.
        MUST get called in order to ensure that Jacobian is build up properly
        '''
        self.row_ptr += self.n_fill
        if self.filling_in_progress == False:
            raise RuntimeError
        self.filling_in_progress = False
        
        
    def fill(self, constr_type, row, content):
        '''
        Handle to specified part of the Jacobian

        Parameters
        ----------
        constr_type : char, type of optimization variable. 'c', 'p' or 'u'
        row : int, number of row to fill (relative to row pointer)
        content: array the jacobian should get filled with

        Returns
        -------
        Handle to specified part of Jacobian (np.ndarray of size (1xn_constraints))

        '''
        if self.filling_in_progress == False:
            raise RuntimeError("Jacobian was not prepared to be filled")
        assert(row <= self.n_fill)
        if constr_type == 'c':
            col_start = 0
            col_end   = self.n_w_c
        elif constr_type == 'p':
            col_start = self.n_w_c
            col_end   = self.n_w_c + self.n_w_p
        elif constr_type == 'u':
            col_start = self.n_w_c + self.n_w_p
            col_end   = self.n_optvar
        else:
            raise ValueError("wrong constr_type" + str(constr_type))
        assert(len(content) == col_end-col_start), (len(content), col_end-col_start)
        if np.sum(content) == 0:
            print("[Jacobian.py] warning! empty row:", constr_type, row)
        self.jac[self.row_ptr + row, col_start:col_end] = content
    
    
    def output(self):
        '''return jacobian in ravel format'''
        for i_r in range(self.jac.shape[0]):
            if np.sum(self.jac[i_r, :]) == 0:
                print("[Jacobian.py] warning! empty row at row nr. ", i_r)
        return np.ravel(self.jac)
        
    
        

        
        