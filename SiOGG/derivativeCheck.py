# -*- coding: utf-8 -*-
"""
Script that helps to verify the Gradient and Jacobian of the problem

Derivations are calculated using the central finite difference method:
    
            f(x+h) - f(x-h)
    f'(x) = ---------------
                 2*h
                
Source: https://v8doc.sas.com/sashtml/ormp/chap5/sect28.htm
"""

from problem import Problem
import numpy as np
import matplotlib.pyplot as plt

problem = Problem()


# vector to check derivative at
w0 = np.random.rand(problem.n_optvar)
#w0 = np.zeros(problem.n_optvar)
# h should be as small as machine precision allows:
h = np.sqrt(np.finfo(float).eps)


### comparison of gradients of objective #####################################
# analytical
grad_an = problem.gradient(w0)

# finite difference (numerical)
grad_nu = np.zeros(problem.n_optvar)
for i in range(problem.n_optvar):
    wu, wl = np.copy(w0), np.copy(w0)
    wu[i] += h
    wl[i] -= h
    grad_nu[i] = (problem.objective(wu) - problem.objective(wl))/(2*h)

difference = grad_an - grad_nu

# plot
plt.bar(np.arange(problem.n_optvar), difference)
plt.axvline(problem.n_w_c, color="k", linestyle="--")
plt.axvline(problem.n_w_c+problem.n_w_p, color="k", linestyle="--")
plt.axhline(0, color="k")
plt.grid()
plt.ylim((-1, 1))
plt.title("Difference between analytical and numerical Gradient")
plt.xlabel("number of constraint")
plt.ylabel("absolute difference")
plt.show()

### comparison of Jacobian of constraints ####################################
# analytical
jac_an = problem.jacobian(w0).reshape((problem.n_constr, problem.n_optvar))

# finite difference (numerical)
jac_nu = np.zeros((problem.n_optvar, problem.n_constr))
for i in range(problem.n_optvar):
    wu, wl = np.copy(w0), np.copy(w0)
    wu[i] += h
    wl[i] -= h
    jac_nu[i] = (problem.constraints(wu) - problem.constraints(wl))/(2*h)
jac_nu = jac_nu.T

difference = jac_an - jac_nu
mat = plt.matshow(difference)
plt.colorbar(mat)
plt.axvline(problem.n_w_c, color="k", linestyle="--")
plt.axvline(problem.n_w_c+problem.n_w_p, color="k", linestyle="--")
hlines = np.cumsum([problem.comSplineContinuity.amount(),
                    problem.comIniFiPos.amount(),
                    problem.systemDynamics.amount()])
for h in hlines:
    plt.axhline(h, color="k", linestyle="--")
plt.title("Difference between analytical and numerical Jacobian")
plt.show()
    
# zoom:
mat = plt.matshow(difference[hlines[0]:, problem.n_w_c:])
plt.colorbar(mat)
plt.axvline(problem.n_w_p, color="k", linestyle="--")
plt.show()
    
    
    