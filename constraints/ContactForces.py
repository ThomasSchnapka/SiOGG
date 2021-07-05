"""
If foot is not in contact, its weighting factor must be = 0
"""

import numpy as np

class ContactForces:
    '''
    Constraint regarding unilateral forces
    '''
    def __init__(self, problem):
        self.n_s = problem.n_s
        self.n_c = problem.n_c
        self.n_f = problem.n_f
        self.n_w_non_u = problem.n_w_c + problem.n_w_p
        self.n_optvar = problem.n_optvar
        self.cont_seq = problem.cont_seq
        
        
    def constraint(self, optvar):
        '''
        TODO
        '''
        d = np.zeros(self.n_s*self.n_c*3*self.n_f)
        for i_s in range(self.n_s):
            for i_c in range(self.n_c):
                for i_u in range(3):
                    vertex_weight = optvar.vertexweight.get_lambda((i_s*self.n_c + i_c), i_u)
                    for i_f in range(self.n_f):
                        d[i_s*self.n_c*3 + i_c*3 + i_u + i_f] = (1-self.cont_seq[i_s, i_f])*vertex_weight[i_f]
        return d
    
    def jacobian(self, optvar):
        #jac = np.zeros((self.n_s*self.n_c*3, self.n_optvar))
        #for i_c in range(self.n_c*self.n_s):
        #   for i_u in range(3):
        #       idx = i_c*3 + i_u
        #       jac[self.n_w_non_u + idx*self.n_f:self.n_w_non_u + (idx+1)*self.n_f] = 1
        #return jac
        return None
        
    
    
    def amount(self):
        return self.n_s*self.n_c*3*self.n_f