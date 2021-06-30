# -*- coding: utf-8 -*-


import numpy as np
import cyipopt
from scipy.linalg import block_diag

from CostFunction import CostFunction
from c_COMSplineContinuity import c_COMSplineContinuity
from c_COMIniFiPos import c_COMIniFiPos
from c_SystemDynamics import c_SystemDynamics
from c_RangeOfMotion import c_RangeOfMotion

### default parameters ######################################################

# geometry
r = 0.02    # half of edge length of RoM
h = 0.50      # hight of COM
p_nom = np.array([[0.2, -0.2],
                  [  0,    0]])

# contact sequence
#c = np.array([[1, 1, 1, 1],
#              [1, 1, 1, 0],
#              [1, 1, 1, 1]])

c = np.array([[1, 1],
              [1, 0],
              [1, 1]])


# paramters
n_s = c.shape[0]    # number of steps
n_f = 2# 4     # number of feet
n_u = 9    # number of lambda (vertex weights) PER STEP
n = 3   # number of quartic polynomials describing the COM motion PER STEP

T_s = 1   # time per step

# initial conditions
x_com_0 = np.array([[0.1, 0.0],       # [dimension][derivative]
                    [0.0, 0.0]])
x_com_T = np.array([[0.0, 0.0,],       # [dimension][derivative]
                    [0.0, 0.0]])




### problem class ###########################################################

class Problem:
    '''
    Problem formulation
    
    optimization variable: w = [w_c, w_p, w_u]
    '''
    def __init__(self,
                 c=c,
                 n_f=n_f,
                 n_u=n_u,
                 n=n,
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
        self.n_u = n_u    # number of lambda (vertex weights) PER STEP
        self.n = n    # number of quartic polynomials describing the COM motion PER STEP
        
        self.T_s = T_s  # time per step (T must be multiple of T_s)
        
        # rounding is neccessary to prevent errors from wrong decimals 
        # resulting by the base 2 representation
        
        self.T = round(self.n_s*self.T_s, 9)       # total horizon time
        self.T_u = round(self.T_s/self.n_u, 9)     # time interval for lambda (vertex weights)
        self.T_c = round(self.T_s/self.n, 9)       # time spend in each COM spline
        
        self.n_w_c = int(2*5*n*self.n_s)
        self.n_w_p = int(2*n_f*self.n_s)
        self.n_w_u = int(n_f*n_u*n_s)
        self.n_optvar = self.n_w_c + self.n_w_p +  self.n_w_u   # length of optimization vector
        
        
        # initial conditions
        self.x_com_0 = x_com_0
        self.x_com_T = x_com_T
        
        
        # cost function
        self.costFunction = CostFunction(self)
        
        # constraints
        self.comSplineContinuity = c_COMSplineContinuity(self)
        self.comIniFiPos = c_COMIniFiPos(self)
        self.systemDynamics = c_SystemDynamics(self)
        self.rangeOfMotion = c_RangeOfMotion(self)
        self.n_constr = 0
        #self.n_constr = self.comSplineContinuity.amount()  # number of constraints
        #self.n_constr += self.comIniFiPos.amount() 
        #self.n_constr += self.systemDynamics.amount() 
        self.n_constr += self.rangeOfMotion.amount() 
        
        # ensure correct dimension of input data
        assert(self.p_nom.shape[0] == self.n_f)
        assert(self.c.shape[1] == self.n_f)
        
        
        
        
    def objective(self, w):
        '''The callback for calculating the objective'''
        #return np.sum(w)
        return self.costFunction.cost(w)


    def gradient(self, w):
        '''The callback for calculating the gradient'''
        return self.costFunction.gradient(w)


    def constraints(self, w):
        '''The callback for calculating the constraints'''
        #cons = np.zeros((self.n_constr, self.n_optvar))
        #cons = np.hstack((#self.comSplineContinuity.constraint(w),
         #                 self.comIniFiPos.constraint(w),
                          #self.systemDynamics.constraint(w),
          #                self.rangeOfMotion.constraint(w)
           #               ))
        cons = self.rangeOfMotion.constraint(w)
        #cons = self.systemDynamics.constraint(w)
        print("cost", np.around(self.costFunction.cost(w), 3),
              "cons", np.around(np.sum(np.square(cons)), 3))
        return cons


    def jacobian(self, w):
        '''The callback for calculating the Jacobian'''
        #jac = np.vstack((#self.comSplineContinuity.jacobian(w),
        #                 self.comIniFiPos.jacobian(w),
        #                 #self.systemDynamics.jacobian(w),
        #                 self.rangeOfMotion.jacobian(w)
        #                  ))
        jac = self.rangeOfMotion.jacobian(w)
        #ac = self.systemDynamics.jacobian(w)
        jac = np.ravel(jac)
        return jac
    
    
    def solve(self):
        '''construct and solve IPOPT problem'''
        #
        # Define the problem
        #
        print("[problem.py] constructing problem")
        w0 = np.random.rand(self.n_optvar)
        w0[self.n_w_c:-self.n_w_u] = 0
        #w0 = np.zeros(self.n_optvar)
    
        lb = np.ones(self.n_optvar)*-9999
        ub = np.ones(self.n_optvar)*9999
    
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
        #nlp.addOption('derivative_test', 'second-order')
        #nlp.add_option('mu_strategy', 'adaptive')
        nlp.add_option('tol', 1e-5)
        nlp.add_option('max_iter', 50)
    
        #
        # Scale the problem (Just for demonstration purposes)
        #
        #nlp.set_problem_scaling(
        #    obj_scaling=2,
        #    x_scaling=[1, 1, 1, 1]
        #    )
        #nlp.add_option('nlp_scaling_method', 'user-scaling')
    
        #
        # Solve the problem
        #
        print("[problem.py] solving problem")
        w, info = nlp.solve(w0)
        print("[problem.py] found optimal solution!")
        return w
        
        
        
if __name__ == '__main__':
    # test
    problem = Problem()
    w = problem.solve()
    
    #'''
    import matplotlib.pyplot as plt
    from CenterOfMass import CenterOfMass
    from CenterOfPressure import CenterOfPressure
    from OptvarFootPos import OptvarFootPos
    from OptvarVertexWeight import OptvarVertexWeight
    
    plt.figure(figsize=(10, 10))
    plt.title("COM and COP")
    n_hlines = int(problem.T/problem.T_c)
    for i in range(n_hlines):
        plt.axvline(i*problem.T_c, linestyle="--", color="k")
    plt.axhline(0, linestyle="--", color="k")
    
    # plot COM
    n_eval = 50
    com = CenterOfMass(problem)
    tsteps = np.linspace(0, problem.T, n_eval)
    com_x = np.zeros(n_eval)
    com_y = np.zeros(n_eval)
    for i in range(n_eval):
        com_x[i] = com.get_c(w, tsteps[i], "x")
        com_y[i] = com.get_c(w, tsteps[i], "y")
    plt.plot(tsteps, com_x, "o-", label="com_x")
    plt.plot(tsteps, com_y, "o-", label="com_y")
    
    # plot COP
    cop = CenterOfPressure(problem)
    n_eval = problem.n*problem.n_s*3
    tsteps = np.linspace(0, problem.T, n_eval)
    cop_x = np.zeros(n_eval)
    cop_y = np.zeros(n_eval)
    for i in range(problem.n*problem.n_s):
        cop_x[3*i+0] = cop.get_u(w,             0, "x", i)
        cop_x[3*i+1] = cop.get_u(w, problem.T_c/2, "x", i)
        cop_x[3*i+2] = cop.get_u(w,   problem.T_c, "x", i)
        cop_y[3*i+0] = cop.get_u(w,             0, "y", i)
        cop_y[3*i+1] = cop.get_u(w, problem.T_c/2, "y", i)
        cop_y[3*i+2] = cop.get_u(w,   problem.T_c, "y", i)
    plt.plot(tsteps, cop_x, "v-", label="cop_x")
    plt.plot(tsteps, cop_y, "v-", label="cop_y")
    
    # plot foot location
    ofp = OptvarFootPos(w, problem)
    tsteps=np.linspace(0, problem.T, problem.n_s)
    foot_pos_x = np.zeros((problem.n_s, problem.n_f))
    foot_pos_y = np.zeros((problem.n_s, problem.n_f))
    for i in range(problem.n_s):
        foot_pos_x[i] = ofp.get_feet_pos(i, "x")
        foot_pos_y[i] = ofp.get_feet_pos(i, "y")
        
    for i in range(problem.n_f):
        plt.plot(tsteps, foot_pos_x[:,i], "o", markersize=10, label=(str(i)+ "_x"))
        plt.plot(tsteps, foot_pos_y[:,i], "v", markersize=10, label=(str(i)+ "_y"))
    
    
    
    plt.legend()
    plt.ylim((-0.5, 0.5))
    plt.show()
    
    # plot lambda values
    plt.figure(figsize=(10, 5))
    plt.title("weights")
    ovw = OptvarVertexWeight(w, problem)
    weights = np.zeros((problem.n_u*problem.n_s, problem.n_f))
    for i in range(problem.n_u*problem.n_s):
        weights[i] = ovw.get_lambda_u(i)
        
    for i in range(problem.n_f):
        plt.plot(weights[:,i], label=str(i))
    
    plt.legend()
    plt.show()
    
    
    # for evaluating the jacobian
    #jac = problem.jacobian(w)
    #jac = np.reshape(jac, (int(len(jac)/problem.n_optvar), problem.n_optvar))
    
    
    
    


