# -*- coding: utf-8 -*-
"""
manages optimization variable

w = [w_c, w_p, w_u]
"""


import numpy as np

from CenterOfMass import CenterOfMass
from FootPos import FootPos
from VertexWeight import VertexWeight
from CenterOfPressure import CenterOfPressure

class OptVar:
    '''
    Object holding the optimization variable
    '''
    def __init__(self, problem, w):
        
        self.w_c = w[:problem.n_w_c]
        self.w_p = w[problem.n_w_c:-problem.n_w_u]
        self.w_u = w[-problem.n_w_u:]
        
        self.com = CenterOfMass(problem, self.w_c)
        self.footpos = FootPos(problem, self.w_p)
        self.vertexweight = VertexWeight(problem, self.w_u)
        self.cop = CenterOfPressure(problem, self.footpos, self.vertexweight)