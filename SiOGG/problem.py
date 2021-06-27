# -*- coding: utf-8 -*-


import numpy as np
import cyipopt
from scipy.linalg import block_diag

from CostFunction import CostFunction
from c_COMSplineContinuity import c_COMSplineContinuity
from c_COMIniFiPos import c_COMIniFiPos
from c_SystemDynamics import c_SystemDynamics

### default parameters ######################################################

# geometry
r = 0.02    # half of edge length of RoM
h = 20      # hight of COM
p_nom = np.array([[ 20, -10],
                  [ 20,  10],
                  [-20, -10],
                  [-20,  10]])

# contact sequence
c = np.array([[1, 1, 1, 1],
              [1, 1, 1, 1],
              [1, 1, 1, 1]])


# paramters
n_s = c.shape[0]    # number of steps
n_f = 4     # number of feet
n_u = 9    # number of lambda (vertex weights) PER STEP
n = 3   # number of quartic polynomials describing the COM motion PER STEP

T_s = 0.5   # time per step (T must be multiple of T_s)
support_ratio = 0.8

# initial conditions
x_com_0 = np.array([[10, 100,],       # [dimension][derivative]
                    [-10, 0]])
x_com_T = np.array([[-10, 0,],       # [dimension][derivative]
                    [10, 0]])

p_legs = np.array([[[1, 0],
                 [1, 9]],
         
                [[1, 2],
                 [1, 2]],
         
                [[1, 2],
                 [1, 2]],
         
                [[0, 0],
                 [0, 0]]])


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
                 support_ratio=support_ratio,
                 x_com_0=x_com_0,
                 x_com_T=x_com_T,
                 p_legs=p_legs,
                 r=r,
                 h=h,
                 p_nom=p_nom,
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
        self.support_ratio = support_ratio
        
        self.T = self.n_s*self.T_s       # total horizon time
        self.T_u = self.T/self.n_u       # time interval for lambda (vertex weights)
        self.T_c = self.T/self.n  # time spend in each COM spline
        
        self.n_w_c = 2*5*self.n*self.n_s
        self.n_w_p = 2*self.n_s*n_f
        self.n_w_u = n_f*n_u*n_s
        self.n_optvar = self.n_w_c + self.n_w_p +  self.n_w_u   # length of optimization vector
        
        
        # initial conditions
        self.x_com_0 = x_com_0
        self.x_com_T = x_com_T
        self.p_legs = p_legs
        
        
        # cost function
        self.costFunction = CostFunction(self)
        
        # constraints
        self.comSplineContinuity = c_COMSplineContinuity(self)
        self.comIniFiPos = c_COMIniFiPos(self)
        self.systemDynamics = c_SystemDynamics(self)
        self.n_constr = self.comSplineContinuity.amount()  # number of constraints
        self.n_constr += self.comIniFiPos.amount() 
        self.n_constr += self.systemDynamics.amount() 
        
        
        
        
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
        cons = np.hstack((self.comSplineContinuity.constraint(w),
                          self.comIniFiPos.constraint(w),
                          self.systemDynamics.constraint(w)
                          ))
        print("cost", np.around(self.costFunction.cost(w), 2),
              "cons", np.around(sum(cons), 2))
        return cons


    def jacobian(self, w):
        '''The callback for calculating the Jacobian'''
        jac = np.vstack((self.comSplineContinuity.jacobian(w),
                         self.comIniFiPos.jacobian(w),
                         self.systemDynamics.jacobian(w)
                         ))
        jac = np.ravel(jac)
        return jac
    
    
    def solve(self):
        '''construct and solve IPOPT problem'''
        #
        # Define the problem
        #
        print("[problem.py] constructing problem")
        x0 = np.random.rand(self.n_optvar)
        #x0 = np.zeros(self.n_optvar)
    
        lb = np.ones(self.n_optvar)*-99
        ub = np.ones(self.n_optvar)*99
    
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
        nlp.add_option('max_iter', 8)
    
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
        w, info = nlp.solve(x0)
        print("[problem.py] found optimal solution!")
        return w
        
        
        
if __name__ == '__main__':
    # test
    problem = Problem()
    w = problem.solve()
    
    #'''
    import matplotlib.pyplot as plt
    from CenterOfMass import CenterOfMass
    
    plt.figure(figsize=(10, 10))
    n_hlines = int(problem.T/(problem.T_c*problem.n))
    for i in range(n_hlines):
        plt.axvline(i*problem.T_c, linestyle="--", color="k")
        
    n_eval = 50
    com = CenterOfMass(problem)
    tsteps = np.linspace(0, problem.T, n_eval)
    x = np.zeros(n_eval)
    y = np.zeros(n_eval)
    for i in range(n_eval):
        x[i] = com.get_c(w, tsteps[i], "x")
        y[i] = com.get_c(w, tsteps[i], "y")
    plt.plot(tsteps, x, "o-")
    plt.plot(tsteps, y, "o-")
    plt.show()
    #'''
    #splineCOM = SplineCOM(problem)
    #print(splineCOM.get_dcx_dw(w, problem.T_c, 2))
    
    
    # for evaluating the jacobian
    #jac = problem.jacobian(w)
    #jac = np.reshape(jac, (int(len(jac)/problem.n_optvar), problem.n_optvar))
    
    
    
    


