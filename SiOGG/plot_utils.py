# -*- coding: utf-8 -*-
"""
plot results of SiOGG optimization
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_results(problem, optvar, plot_combined=False):
    '''plot results'''
    
    # plot x/y-diagramm ######################################################
    if plot_combined:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        fig.suptitle("COM and COP trajectory", y=0.93, size=24)
        
        # plot lines indicating spline segments
        n_hlines = int(problem.T/problem.T_c)
        for i in range(n_hlines+2):
            ax.axvline(i*problem.T_c, linestyle="--", color="k", lw=0.5)
            ax.axhline(i*problem.T_c, linestyle="--", color="k", lw=0.5)
        ax.axhline(0, linestyle="--", color="k", lw=0.5)
        ax.axvline(0, linestyle="--", color="k", lw=0.5)
        
        # plot COM
        n_eval = 50
        tsteps = np.linspace(0, problem.T, n_eval)
        com_x = np.zeros(n_eval)
        com_y = np.zeros(n_eval)
        for i in range(n_eval):
            com_x[i] = optvar.com.get_c(tsteps[i], "x")
            com_y[i] = optvar.com.get_c(tsteps[i], "y")
        ax.plot(com_x, com_y,  marker="o", label="COM", color="tab:blue")
        
        # plot COP
        n_eval = problem.n_c*problem.n_s*3
        tsteps = np.linspace(0, problem.T, n_eval)
        cop_x = np.zeros(n_eval)
        cop_y = np.zeros(n_eval)
        for i_c in range(problem.n_c*problem.n_s):
            for i_u in range(3):
                cop_x[3*i_c+i_u] = optvar.cop.get_cop(i_c, i_u, "x")
                cop_y[3*i_c+i_u] = optvar.cop.get_cop(i_c, i_u, "y")
                
        ax.plot(cop_x, cop_y, linestyle="--", marker="v", label="COP", 
                 color="tab:cyan", alpha=0.8)
        
        # plot foot location
        tsteps=np.linspace(0, problem.T, problem.n_s)
        foot_pos_x = np.zeros((problem.n_s, problem.n_f))
        foot_pos_y = np.zeros((problem.n_s, problem.n_f))
        for i in range(problem.n_s):
            foot_pos_x[i] = optvar.footpos.get_foot_pos(i, "x")
            foot_pos_y[i] = optvar.footpos.get_foot_pos(i, "y")
            
        for i in range(0, problem.n_f):
            rgba = np.ones((problem.n_s,4))
            rgba[:,:3] = (i/problem.n_f)
            rgba[problem.cont_seq[:, i]==0, 3] = 0.1;
            ax.scatter(foot_pos_x[:,i], foot_pos_y[:,i], marker=">", s=150, 
                        label=("leg " + str(i)), cmap="viridis", color=rgba)
            
        # plot range of motion
        for i_f in range(0, problem.n_f):
            for i_s in range(0, problem.n_s):
                com_x = optvar.com.get_c(tsteps[i_s], "x")
                com_y = optvar.com.get_c(tsteps[i_s], "y")
                p_x = problem.p_nom[0, i_f] - problem.r/2 + com_x
                p_y = problem.p_nom[1, i_f] - problem.r/2 + com_y
                ax.add_patch(plt.Rectangle((p_x, p_y), problem.r, problem.r, 
                                            color='k', lw=1, fill=False, alpha=0.5))
                
        # add ROM patches to legend workaround
        ax.plot([],[], color='k', lw=1, alpha=0.5, label="ROM")
        
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(bbox_to_anchor=(1.05, 1))
        ax.set_xlim((-0.5, 0.5))
        ax.set_ylim((-0.5, 0.5))
        plt.show()
            
        
    # plot x/t and y/t-diagramm #############################################
    else:
        fig, [axx, axy] = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
        fig.suptitle("COM and COP trajectory", y=0.93, size=24)
        
        # plot lines indicating spline segments
        n_hlines = int(problem.T/problem.T_c)
        for i in range(n_hlines+2):
            axx.axvline(i*problem.T_c, linestyle="--", color="k", lw=0.5)
            axy.axvline(i*problem.T_c, linestyle="--", color="k", lw=0.5)
        axx.axhline(0, linestyle="--", color="k", lw=0.5)
        axy.axhline(0, linestyle="--", color="k", lw=0.5)
        
        # plot COM
        n_eval = 50
        tsteps = np.linspace(0, problem.T, n_eval)
        com_x = np.zeros(n_eval)
        com_y = np.zeros(n_eval)
        for i in range(n_eval):
            com_x[i] = optvar.com.get_c(tsteps[i], "x")
            com_y[i] = optvar.com.get_c(tsteps[i], "y")
        axx.plot(tsteps, com_x,  marker="o", label="COM", color="tab:blue")
        axy.plot(tsteps, com_y,  marker="o", label="COM", color="tab:orange")
        
        # plot COP
        n_eval = problem.n_c*problem.n_s*3
        tsteps = np.linspace(0, problem.T, n_eval)
        cop_x = np.zeros(n_eval)
        cop_y = np.zeros(n_eval)
        for i_c in range(problem.n_c*problem.n_s):
            for i_u in range(3):
                cop_x[3*i_c+i_u] = optvar.cop.get_cop(i_c, i_u, "x")
                cop_y[3*i_c+i_u] = optvar.cop.get_cop(i_c, i_u, "y")
        axx.plot(tsteps, cop_x, linestyle="--", marker="v", label="COP", 
                 color="tab:cyan", alpha=0.8)
        axy.plot(tsteps, cop_y, linestyle="--", marker="v", label="COP", 
                 color="tab:red", alpha=0.8)
        
        # plot foot location
        tsteps=np.linspace(0, problem.T, problem.n_s)
        foot_pos_x = np.zeros((problem.n_s, problem.n_f))
        foot_pos_y = np.zeros((problem.n_s, problem.n_f))
        for i in range(problem.n_s):
            foot_pos_x[i] = optvar.footpos.get_foot_pos(i, "x")
            foot_pos_y[i] = optvar.footpos.get_foot_pos(i, "y")
            
        for i in range(0, problem.n_f):
            rgba = np.ones((problem.n_s,4))
            rgba[:,:3] = (i/problem.n_f)
            rgba[problem.cont_seq[:, i]==0, 3] = 0.1;
            axx.scatter(tsteps, foot_pos_x[:,i], marker=">", s=150, 
                        label=("leg " + str(i)), cmap="viridis", color=rgba)
            axy.scatter(tsteps, foot_pos_y[:,i], marker=">", s=150, 
                        label=("leg " + str(i)), cmap="viridis", color=rgba)
            
        # plot range of motion
        for i_f in range(0, problem.n_f):
            for i_s in range(0, problem.n_s):
                # x-Dimension
                com_x = optvar.com.get_c(tsteps[i_s], "x")
                px_y = problem.p_nom[0, i_f] - problem.r/2 + com_x
                px_x = tsteps[i_s] - problem.r/2
                axx.add_patch(plt.Rectangle((px_x, px_y), problem.r, problem.r, 
                                            color='k', lw=1, fill=False, alpha=0.5))
                # y-Dimension
                com_y = optvar.com.get_c(tsteps[i_s], "y")
                py_y = problem.p_nom[1, i_f] - problem.r/2 + com_y
                py_x = tsteps[i_s] - problem.r/2
                axy.add_patch(plt.Rectangle((py_x, py_y), problem.r, problem.r, 
                                            color='k', lw=1, fill=False, alpha=0.5))
        
        # add ROM patches to legend workaround
        axx.plot([],[], color='k', lw=1, alpha=0.5, label="ROM")
        axy.plot([],[], color='k', lw=1, alpha=0.5, label="ROM")
        
        axx.set_ylabel("x")
        axy.set_ylabel("y")
        axy.set_xlabel("t [s]")
        axx.legend(bbox_to_anchor=(1.05, 1))
        axy.legend(bbox_to_anchor=(1.05, 1))
        axx.set_ylim((-0.5, 0.5))
        axy.set_ylim((-0.5, 0.5))
        plt.show()
                
    
    # plot lambda values #####################################################
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
    
    #mat = plt.matshow(problem.jacobian(w).reshape(problem.n_constr, problem.n_optvar))
    #plt.colorbar(mat)
    #plt.title("Current Jacobian")
    #plt.show()
    


