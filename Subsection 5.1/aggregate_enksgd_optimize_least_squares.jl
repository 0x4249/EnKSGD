using LinearAlgebra
using Printf
using PyPlot
include("../objectiveFunctions.jl")

# Initialize least squares objective function
eigenvalues = 10.0 .^ (range(-2,4,step=0.5))
d = length(eigenvalues)
Q = diagm(eigenvalues)
b = zeros(d)
compositeObjFun = linear_ls(A=Q,b=b)
m = size(compositeObjFun.inner_func_jacob(ones(d)),1)
@show m, d

# Aggregate parameters
total_runs = 30
maxIter = 60
inner_func_eval_counts = zeros(total_runs)
line_search_fail_counts = zeros(total_runs)
all_line_search_fails = zeros(maxIter,total_runs)
all_phi_vals = zeros(maxIter+1,total_runs)
all_x_vals = zeros(maxIter+1,d,total_runs)

# Print out information?
verbose = false

# Plotting setup
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["text.usetex"] = true
rcParams["figure.autolayout"] = true
color_map = get_cmap("twilight_shifted")

# Repeated algorithm runs
for run in 1:total_runs
    global all_phi_vals, all_x_vals, all_line_search_fails, maxIter
    
    # Optimization setup
    x_0 = 1e5*ones(d)
    K = 20
    sigma = 1e-2
    delta = 1e0
    beta = 1e-8
    Pi_K = I - (1/K)*ones(K,K)
    x_bar = deepcopy(x_0)
    X = x_bar .+ sigma*randn(d,K)*Pi_K
    Y = X*Pi_K
    
    # Backtracking line search parameters
    dt_0 = 1e0
    c_1 = 1e-4
    tau = 0.1
    max_backtracks = 15
    
    # Allocate storage
    line_search_fail_count = 0
    inner_func_eval_count = 0
    outer_func_eval_count = 0
    outer_func_grad_eval_count = 0
    outer_func_hess_eval_count = 0
    phi_vals = zeros(maxIter+1)
    x_vals = zeros(maxIter+1,d)
    line_search_fails = zeros(maxIter)
    
    # Norm clipping
    Y_max = 1e4
    Y_min = 1e-4
    
    # Eigenvalue shift 
    eps = 1e-7
    
    # Evaluate forward map at mean of particle locations
    G_x_bar = compositeObjFun.inner_func_eval(x_bar)
    inner_func_eval_count += 1
    
    # Evaluate loss function
    D_x_bar = compositeObjFun.outer_func_eval(G_x_bar)
    outer_func_eval_count += 1
    
    # Evaluate objective function
    phi_x_bar = D_x_bar
    
    # Store evaluation
    x_vals[1,:] = x_bar
    phi_vals[1] = phi_x_bar
    
    # EnKSGD Loop
    for i in 1:maxIter
        
        # Evaluate forward map at particle locations
        G = zeros(m,K)
        for k in 1:K
            G[:,k] = compositeObjFun.inner_func_eval(X[:,k])
            inner_func_eval_count += 1
        end
        
        # Calculate forward map deviations
        G_devs = G*Pi_K
        if verbose
            @show norm(G_devs)
        end
        
        # Evaluate loss function gradient
        grad_loss_x_bar = compositeObjFun.outer_func_grad(G_x_bar)
        outer_func_grad_eval_count += 1
        
        # Evaluate loss function Hessian
        H_loss_x_bar = compositeObjFun.outer_func_hess(G_x_bar)
        outer_func_hess_eval_count += 1
        if verbose
            @show norm(H_loss_x_bar)
        end
        
        # Compute objective function gradient projected onto particles
        q_loss = G_devs'*grad_loss_x_bar
        q = q_loss
        
        # Prepare for backtracking line search
        dt_prop = dt_0
        ls_flag = false
        line_search_fail = 0
        
        # Initialize variables outside of while loop scope
        dt = 0
        x_bar_new = zeros(d)
        eigvecs = zeros(K,K)
        eigvals = zeros(K)
        G_x_bar_new = zeros(m)
        D_x_bar_new = 0
        phi_x_bar_new = 0
        
        # Backtracking line search for ensemble mean update
        while ls_flag == false
            
            if line_search_fail >= max_backtracks
                dt = 0
                x_bar_new = x_bar
                eigvecs = I
                eigvals = ones(K)
                
                G_x_bar_new = compositeObjFun.inner_func_eval(x_bar_new)
                inner_func_eval_count += 1
                
                D_x_bar_new = compositeObjFun.outer_func_eval(G_x_bar_new)
                outer_func_eval_count += 1
                
                phi_x_bar_new = D_x_bar_new
                
                break
            end
            
            # Compute inverse of proposed transform matrix
            scaled_dt_prop = dt_prop/delta
            T_inv = I + (scaled_dt_prop/K)*G_devs'*H_loss_x_bar*G_devs
            if verbose
                @show T_inv
                @show norm(T_inv - T_inv'), scaled_dt_prop
            end
            
            # Compute eigendecomposition of inverse of proposed transform matrix
            eigvecs_prop, eigvals_prop, ~ = svd(T_inv)
            if verbose
                @show eigvals_prop
            end
            
            # Regularize by shifting eigenvalues by small positive number
            eigvals_prop_shifted = eigvals_prop .+ eps
            if verbose
                @show eigvals_prop_shifted
            end
            
            # Compute proposed transform matrix
            T_prop = eigvecs_prop*diagm(1 ./ eigvals_prop_shifted)*eigvecs_prop'
            
            # Compute proposed upate of ensemble mean
            r_prop = (scaled_dt_prop/K)*T_prop*q
            x_bar_prop = x_bar - Y*r_prop
            
            # Evaluate forward map at proposed mean of particle locations
            G_x_bar_prop = compositeObjFun.inner_func_eval(x_bar_prop)
            inner_func_eval_count += 1
            
            # Evaluate loss function
            D_x_bar_prop = compositeObjFun.outer_func_eval(G_x_bar_prop)
            outer_func_eval_count += 1
            
            # Evaluate objective function
            phi_x_bar_prop = D_x_bar_prop
            
            # Check approximate sufficient decrease condition
            if phi_x_bar_prop <= phi_x_bar - c_1*dot(q, r_prop) && !isnan(phi_x_bar_prop)
                ls_flag = true
                dt = dt_prop
                x_bar_new = x_bar_prop
                eigvecs = eigvecs_prop
                eigvals = eigvals_prop_shifted
                G_x_bar_new = G_x_bar_prop
                D_x_bar_new = D_x_bar_prop
                phi_x_bar_new = phi_x_bar_prop
            else
                dt_prop = tau*dt_prop
                line_search_fail += 1
                if verbose
                    @show dt_prop
                end
            end                                           
        end
        
        # Compute particle deviations update
        T_up_sqrt = eigvecs*diagm(1 ./ sqrt.(eigvals))*eigvecs'
        Y_new = exp(dt/2)*Y*T_up_sqrt + sqrt(beta*delta*dt)*randn(d,K)
        
        # Clip particle deviations
        for k in 1:K
            norm_Y_new_k = norm(Y_new[:,k])
            if norm_Y_new_k > (Y_max)*d
                Y_new[:,k] = (Y_max)*Y_new[:,k]/norm_Y_new_k
            elseif norm_Y_new_k < (Y_min)*d
                Y_new[:,k] = (Y_min)*Y_new[:,k]/norm_Y_new_k
            end
        end
        
        # Update the particle locations                
        X_new = x_bar_new*ones(K)' + Y_new*Pi_K
        
        line_search_fail_count += line_search_fail
        line_search_fails[i] = line_search_fail
        
        if verbose
            @show(line_search_fail)
            @show(inner_func_eval_count,outer_func_eval_count)
            @show(phi_x_bar_new,phi_x_bar)
            @show norm(x_bar_new - x_bar)
        end
        
        x_bar = x_bar_new
        X = X_new
        Y = Y_new 
        G_x_bar = G_x_bar_new
        D_x_bar = D_x_bar_new
        phi_x_bar = phi_x_bar_new
        
        phi_vals[i+1] = phi_x_bar
        x_vals[i+1,:] = x_bar
        
        if verbose
            @show norm(Y)
            @show rank(Y)
        end
        @show run, i
    end
    
    if verbose
        @show(line_search_fail_count)
    end
    
    inner_func_eval_counts[run] = inner_func_eval_count
    line_search_fail_counts[run] = line_search_fail_count
    all_line_search_fails[:,run] = line_search_fails
    all_phi_vals[:,run] = phi_vals
    all_x_vals[:,:,run] = x_vals
end

# Optimality gap figure
pltIter = maxIter
figure()
axes_opt_gap = plt.gca()
axes_opt_gap.set_xlim([0,pltIter])
axes_opt_gap.set_ylim([-20,20])
axes_opt_gap.grid(true)
axes_opt_gap.tick_params(axis="x", labelsize=26)
axes_opt_gap.tick_params(axis="y", labelsize=26)
xlabel(L"Iteration $n$", fontsize=26)
ylabel(L"$\log_{10} \bigg ( \Phi(\mathbf{\bar{x}}_{n}) \bigg )$", fontsize=26)
title("EnKSGD No Noise", fontsize=26)
for run in 1:total_runs
    axes_opt_gap.plot(0:pltIter,log.(10,all_phi_vals[:,run]), alpha=0.8, color=color_map(run/total_runs), label="EnKSGD")
end
axes_opt_gap.text(-0.3, 1.1, "(A)", size=20, weight="bold", transform=axes_opt_gap.transAxes)

