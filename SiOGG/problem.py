# -*- coding: utf-8 -*-


import numpy as np
import cyipopt
from scipy.linalg import block_diag


from constraints.COMSplineContinuity import COMSplineContinuity
from constraints.COMIniFiPos import COMIniFiPos
from constraints.SystemDynamics import SystemDynamics
from constraints.RangeOfMotion import RangeOfMotion
from constraints.UnilateralForces import UnilateralForces

from CostFunction import CostFunction
from Jacobian import Jacobian
from OptVar import OptVar

import warnings
warnings.simplefilter("error", np.VisibleDeprecationWarning)


### default parameters ######################################################

# geometry
r = 0.1       # half of edge length of RoM
h = 0.50      # hight of COM
p_nom = np.array([[0.3, -0.3],
                  [  0,    0]])

# contact sequence
#c = np.array([[1, 1, 1, 1],
#              [1, 1, 1, 0],
#              [1, 1, 1, 1]])

c = np.array([[1, 1],
              [1, 1],
              [1, 1]])


# paramters
n_s = c.shape[0]    # number of steps
n_f = 2# 4     # number of feet
#n_u = 9    # number of lambda (vertex weights) PER STEP
n_c = 3   # number of quartic polynomials describing the COM motion PER STEP

T_s = 0.8   # time per step

# initial conditions
x_com_0 = np.array([[0.1, 0.0],       # [dimension][derivative]
                    [0.0, 0.0]])
x_com_T = np.array([[0.0, 0.0,],       # [dimension][derivative]
                    [0.1, 0.0]])




### problem class ###########################################################

class Problem:
    '''
    Problem formulation
    
    optimization variable: w = [w_c, w_p, w_u]
    '''
    def __init__(self,
                 c=c,
                 n_f=n_f,
                 #n_u=n_u,
                 n_c=n_c,
                 T_s=T_s,
                 x_com_0=x_com_0,
                 x_com_T=x_com_T,
                 p_nom=p_nom,
                 r=r,
                 h=h
                 ):
        
        # geometry
        self.r = r
        self.h = h
        self.p_nom = p_nom
        
        # contact sequence
        self.c = c
        
        # paramters
        self.n_s = c.shape[0]    # number of steps
        self.n_f = n_f     # number of feet
        #self.n_u = n_u    # number of lambda (vertex weights) PER STEP
        self.n_c = n_c    # number of quartic polynomials describing the COM motion PER STEP
        
        self.T_s = T_s  # time per step (T must be multiple of T_s)
        
        # rounding is neccessary to prevent errors from wrong decimals 
        # resulting by the base 2 representation
        
        self.T = round(self.n_s*self.T_s, 9)       # total horizon time
        #self.T_u = round(self.T_s/self.n_u, 9)     # time interval for lambda (vertex weights)
        self.T_c = round(self.T_s/self.n_c, 9)       # time spend in each COM spline
        
        self.n_w_c = int(2*5*n_c*self.n_s)
        self.n_w_p = int(2*n_f*self.n_s)
        self.n_w_u = int(n_f*n_s*3)
        self.n_optvar = self.n_w_c + self.n_w_p +  self.n_w_u   # length of optimization vector
        
        
        # initial conditions
        self.x_com_0 = x_com_0
        self.x_com_T = x_com_T
        
        
        # cost function
        self.costFunction = CostFunction(self)
        
        # constraints
        self.comSplineContinuity = COMSplineContinuity(self)
        self.comIniFiPos = COMIniFiPos(self)
        self.systemDynamics = SystemDynamics(self)
        self.rangeOfMotion = RangeOfMotion(self)
        self.unilateralForces = UnilateralForces(self)
        
        self.n_constr = 0
        self.n_constr += self.comSplineContinuity.amount()  # number of constraints
        self.n_constr += self.comIniFiPos.amount() 
        self.n_constr += self.systemDynamics.amount() 
        #self.n_constr += self.rangeOfMotion.amount() 
        #self.n_constr += self.unilateralForces.amount()
        
        # ensure correct dimension of input data
        assert(self.p_nom.shape[0] == self.n_f)
        assert(self.c.shape[1] == self.n_f)
        
        
        
        
    def objective(self, w):
        '''The callback for calculating the objective'''
        assert(len(w) == self.n_optvar)
        #return w@w
        optvar = OptVar(self, w)
        return self.costFunction.cost(optvar)


    def gradient(self, w):
        '''The callback for calculating the gradient'''
        optvar = OptVar(self, w)
        return self.costFunction.gradient(optvar)
        #return 2*w


    def constraints(self, w):
        '''The callback for calculating the constraints'''
        #cons = np.zeros((self.n_constr, self.n_optvar))
        optvar = OptVar(self, w)
        cons = np.hstack((self.comSplineContinuity.constraint(optvar),
                          self.comIniFiPos.constraint(optvar),
                          self.systemDynamics.constraint(optvar),
        #                  self.rangeOfMotion.constraint(optvar),
        #                  self.unilateralForces.constraint(optvar)
                          ))
        #cons = self.comIniFiPos.constraint(optvar)
        #cons = self.comSplineContinuity.constraint(optvar)
        assert(len(cons) == self.n_constr)
        #cons = self.systemDynamics.constraint(w)
        #a = np.sum(np.square(self.comSplineContinuity.constraint(optvar)))
        #b = np.sum(np.square(self.comIniFiPos.constraint(optvar)))
        #c = np.sum(np.square(self.systemDynamics.constraint(optvar)))
        #d = np.sum(np.square(self.rangeOfMotion.constraint(optvar)))
        #e = np.sum(np.square(self.unilateralForces.constraint(optvar)))
        
        
        #print("cost", np.around(self.costFunction.cost(optvar), 5),
         #     "cons", np.around(np.sum(np.square(cons)), 5))#"|",
             # np.around(a, 3), np.around(b,3), np.around(c,3), np.around(d,3),
              #np.around(e, 3))
        #print(self.comIniFiPos.constraint(w))
        return cons


    def jacobian(self, w):
        '''The callback for calculating the Jacobian'''
        print("jacobian got called")
        optvar = OptVar(self, w)
        jac = Jacobian(self)
        self.comSplineContinuity.fill_jacobian(jac, optvar)
        self.comIniFiPos.fill_jacobian(jac, optvar)
        self.systemDynamics.fill_jacobian(jac, optvar)
        #                 self.rangeOfMotion.jacobian(optvar),
        #                 self.unilateralForces.jacobian(optvar)
        #                  ))
        #jac = self.rangeOfMotion.jacobian(w)
        #jac = self.comIniFiPos.jacobian(w)
        return jac.output()
    
    
    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials):
        """Prints information at every Ipopt iteration."""

        msg = "Objective value at iteration #{:d} is - {:g}"

        print(msg.format(iter_count, obj_value))
    
    
    def solve(self):
        '''construct and solve IPOPT problem'''
        #
        # Define the problem
        #
        print("[problem.py] constructing problem")
        w0 = np.random.rand(self.n_optvar)
        w0[:-self.n_w_u] = 1.0/self.n_f
    
        lb = np.ones(self.n_optvar)*-999
        lb[self.n_w_p:-self.n_w_u] = -10.0
        lb[-self.n_w_u:] = 0.0
        ub = np.ones(self.n_optvar)*999
        ub[self.n_w_p:-self.n_w_u] = 10.0
        ub[-self.n_w_u:] = 1.0
    
        cl = np.zeros(self.n_constr)
        cu = np.zeros(self.n_constr)
    
        nlp = cyipopt.Problem(
            n=self.n_optvar,
            m=self.n_constr,
            problem_obj=self,   # will use itself
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu
            )
    
        #
        # Set solver options
        #
        nlp.add_option('derivative_test', 'first-order')
        nlp.add_option("jacobian_approximation", "finite-difference-values")
        
        #nlp.add_option("bound_relax_factor", 0.0)
        nlp.add_option("max_cpu_time", 30.0)
        #nlp.add_option('mu_strategy', 'adaptive')
        #nlp.add_option('tol', 1e-8)
        #nlp.add_option('max_iter', 50)
    
        
        #
        # Solve the problem
        #
        print("[problem.py] solving problem")
        w, info = nlp.solve(w0)
        print("Status: ", info['status_msg'])
        return w
        
        
        
if __name__ == '__main__':
    # test
    problem = Problem()
    w = problem.solve()
    optvar = OptVar(problem, w)
    
    #'''
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 10))
    plt.title("COM and COP")
    n_hlines = int(problem.T/problem.T_c)
    for i in range(n_hlines+1):
        plt.axvline(i*problem.T_c, linestyle="--", color="k")
    plt.axhline(0, linestyle="--", color="k")
    
    # plot COM
    n_eval = 50
    tsteps = np.linspace(0, problem.T, n_eval)
    com_x = np.zeros(n_eval)
    com_y = np.zeros(n_eval)
    for i in range(n_eval):
        com_x[i] = optvar.com.get_c(tsteps[i], "x")
        com_y[i] = optvar.com.get_c(tsteps[i], "y")
    plt.plot(tsteps, com_x, "o-", label="com_x")
    plt.plot(tsteps, com_y, "o-", label="com_y")
    
    # plot COP
    n_eval = problem.n_c*problem.n_s*3
    tsteps = np.linspace(0, problem.T, n_eval)
    cop_x = np.zeros(n_eval)
    cop_y = np.zeros(n_eval)
    for i_c in range(problem.n_c*problem.n_s):
        for i_u in range(3):
            cop_x[3*i_c+i_u] = optvar.cop.get_cop(i_c, i_u, "x")
            cop_y[3*i_c+i_u] = optvar.cop.get_cop(i_c, i_u, "y")
            
    plt.plot(tsteps, cop_x, "v-", label="cop_x")
    plt.plot(tsteps, cop_y, "v-", label="cop_y")
    
    # plot foot location
    tsteps=np.linspace(0, problem.T, problem.n_s)
    foot_pos_x = np.zeros((problem.n_s, problem.n_f))
    foot_pos_y = np.zeros((problem.n_s, problem.n_f))
    for i in range(problem.n_s):
        foot_pos_x[i] = optvar.footpos.get_foot_pos(i, "x")
        foot_pos_y[i] = optvar.footpos.get_foot_pos(i, "y")
        
    for i in range(problem.n_f):
        plt.plot(tsteps, foot_pos_x[:,i], "o", markersize=10, label=(str(i)+ "_x"))
        plt.plot(tsteps, foot_pos_y[:,i], "v", markersize=10, label=(str(i)+ "_y"))
    
    
    
    plt.legend()
    plt.ylim((-0.5, 0.5))
    plt.show()
    
    # plot lambda values
    plt.figure(figsize=(10, 5))
    plt.title("weights")
    weights = np.zeros((3*problem.n_c*problem.n_s, problem.n_f))
    for i_c in range(problem.n_c*problem.n_s):
        for i_u in range(3):
            weights[3*i_c + i_u] = optvar.vertexweight.get_lambda(i_c, i_u)
        
    for i in range(problem.n_f):
        plt.plot(weights[:,i], label=str(i))
    
    plt.ylim((0, 1))
    plt.legend()
    plt.show()
    
    
    # for evaluating the jacobian
    #jac = problem.jacobian(w)
    #jac = np.reshape(jac, (int(len(jac)/problem.n_optvar), problem.n_optvar))
    
    
    
    


