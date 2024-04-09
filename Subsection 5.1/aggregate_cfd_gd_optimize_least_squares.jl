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

# Function to estimate gradient using standard central finite differences 
function cfd_grad(func_eval,x;h=1e-3)
    d = length(x)
    V_pos = Matrix{Float64}(I,d,d) 
    f_pos = zeros(d)
    f_neg = zeros(d)
    for j in 1:d
        f_pos[j] = func_eval(x + h*V_pos[:,j])
        f_neg[j] = func_eval(x - h*V_pos[:,j])
    end
    cfd_grad = (f_pos - f_neg)/(2*h)
    func_eval_count = 2*d
    return cfd_grad, func_eval_count
end

# Aggregate parameters
total_runs = 30
maxIter = 60
obj_func_eval_counts = zeros(total_runs)
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
    x_bar = deepcopy(x_0)
    h = 1e-4
    
    # Backtracking line search parameters
    dt_0 = 1e0
    c_1 = 1e-4
    tau = 0.1
    max_backtracks = 15
    
    # Allocate storage
    line_search_fail_count = 0
    obj_func_eval_count = 0
    phi_vals = zeros(maxIter+1)
    x_vals = zeros(maxIter+1,d)
    line_search_fails = zeros(maxIter)
    
    phi_x_bar = Inf
    g_x_bar = zeros(d)
    
    # Central Finite Difference (CFD) Gradient Descent Loop
    for i in 1:maxIter
        
        # Initial function evaluation and gradient estimation
        if i == 1
            # Evaluate objective function
            phi_x_bar = compositeObjFun.func_eval(x_bar)
            obj_func_eval_count += 1
            
            # Store objective function evaluation
            x_vals[1,:] = x_bar
            phi_vals[1] = phi_x_bar
            
            # Estimate objective function gradient
            g_x_bar, func_eval_count = cfd_grad(compositeObjFun.func_eval,x_bar,h=h)
            obj_func_eval_count += func_eval_count
        end
        
        # Prepare for backtracking line search
        dt_prop = dt_0
        ls_flag = false
        line_search_fail = 0
        
        # Initialize variables outside of while loop scope
        dt = 0
        x_bar_new = zeros(d)
        phi_x_bar_new = 0
        g_x_bar_new = zeros(d)
        
        # Backtracking line search for ensemble mean update
        while ls_flag == false
            
            if line_search_fail >= max_backtracks
                dt = 0
                x_bar_new = x_bar
                phi_x_bar_new = compositeObjFun.func_eval(x_bar_new)
                
                # Estimate objective function gradient
                g_x_bar_new, func_eval_count = cfd_grad(compositeObjFun.func_eval,x_bar_new,h=h)
                obj_func_eval_count += func_eval_count
                
                break
            end
            
            # Compute proposed upate of ensemble mean
            p = -g_x_bar
            x_bar_prop = x_bar + dt_prop*p
            
            # Evaluate objective function
            phi_x_bar_prop = compositeObjFun.func_eval(x_bar_prop)
            obj_func_eval_count += 1
            
            # Check approximate sufficient decrease condition
            if phi_x_bar_prop <= phi_x_bar + c_1*dt_prop*dot(g_x_bar, p) && !isnan(phi_x_bar_prop)
                ls_flag = true
                dt = dt_prop
                x_bar_new = x_bar_prop
                phi_x_bar_new = phi_x_bar_prop
                
                # Estimate objective function gradient
                g_x_bar_new, func_eval_count = cfd_grad(compositeObjFun.func_eval,x_bar_new,h=h)
                obj_func_eval_count += func_eval_count
            else
                dt_prop = tau*dt_prop
                line_search_fail += 1
                if verbose
                    @show dt_prop
                end
            end                                           
        end
        
        line_search_fail_count += line_search_fail
        line_search_fails[i] = line_search_fail
        
        if verbose
            @show(line_search_fail)
            @show(obj_func_eval_count)
            @show(phi_x_bar_new,phi_x_bar)
            @show norm(x_bar_new - x_bar)
        end
        
        x_bar = x_bar_new
        phi_x_bar = phi_x_bar_new
        g_x_bar = g_x_bar_new
        
        phi_vals[i+1] = phi_x_bar
        x_vals[i+1,:] = x_bar

        @show run, i
    end
    
    if verbose
        @show(line_search_fail_count)
    end
    
    obj_func_eval_counts[run] = obj_func_eval_count
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
title("CFD GD No Noise", fontsize=26)
for run in 1:total_runs
    axes_opt_gap.plot(0:pltIter,log.(10,all_phi_vals[:,run]), alpha=0.8, color=color_map(run/total_runs), label="CFD GD")
end
axes_opt_gap.text(-0.3, 1.1, "(D)", size=20, weight="bold", transform=axes_opt_gap.transAxes)

