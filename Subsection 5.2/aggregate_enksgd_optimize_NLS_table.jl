using LinearAlgebra
using NLSProblems
using Printf
using Statistics
include("../objectiveFunctions.jl")

# Initialize objective function
nls = eval(:nls_rosenbrock)() 
#nls = eval(:hs25)()
#nls = eval(:mgh11)()
#nls = eval(:mgh18)()
#nls = eval(:tp294)()
#nls = eval(:mgh19)()
#nls = eval(:tp296)()
#nls = eval(:mgh22)()
#nls = eval(:tp297)()
#nls = eval(:tp304)()
#nls = eval(:tp305)()
compositeObjFun = nls_objective(nls)
(m,d) = size(compositeObjFun.inner_func_jacob(nls.meta.x0))
@show m, d

# Aggregate parameters
total_runs = 30
max_inner_func_evals = 500
final_obj_vals = zeros(total_runs)
line_search_fail_counts = zeros(total_runs)
inner_func_eval_counts = zeros(total_runs)
outer_func_eval_counts = zeros(total_runs)
iter_counts = zeros(total_runs)

# Print out information?
verbose = false

# Repeated algorithm runs
for run in 1:total_runs
    global max_inner_func_evals, inner_func_eval_counts, outer_func_eval_counts, iter_counts, final_obj_vals
    
    # Optimization setup
    x_0 = nls.meta.x0
    K = 8
    sigma = 1e-2
    delta = 1e-3
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
    final_obj_val = Inf
    iter_count = 0
    
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
    
    # EnKSGD Loop
    break_flag = false
    while !break_flag
        iter_count += 1
        
        # Evaluate forward map at particle locations
        G = zeros(m,K)
        for k in 1:K
            G[:,k] = compositeObjFun.inner_func_eval(X[:,k])
            inner_func_eval_count += 1
            
            if inner_func_eval_count >= max_inner_func_evals
                final_obj_vals[run] = final_obj_val
                line_search_fail_counts[run] = line_search_fail_count
                inner_func_eval_counts[run] = inner_func_eval_count
                outer_func_eval_counts[run] = outer_func_eval_count
                iter_counts[run] = iter_count
                break_flag = true
                break
            end
        end
        
        if break_flag
            break
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
                
                if inner_func_eval_count >= max_inner_func_evals
                    final_obj_vals[run] = final_obj_val
                    line_search_fail_counts[run] = line_search_fail_count
                    inner_func_eval_counts[run] = inner_func_eval_count
                    outer_func_eval_counts[run] = outer_func_eval_count
                    iter_counts[run] = iter_count
                    break_flag = true
                    break
                end
                
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
            
            if inner_func_eval_count >= max_inner_func_evals
                final_obj_vals[run] = final_obj_val
                line_search_fail_counts[run] = line_search_fail_count
                inner_func_eval_counts[run] = inner_func_eval_count
                outer_func_eval_counts[run] = outer_func_eval_count
                iter_counts[run] = iter_count
                break_flag = true
                break
            end
            
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
        
        if break_flag
            break
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
        
        final_obj_val = phi_x_bar_new
        
        if verbose
            @show norm(Y)
            @show rank(Y)
        end
        @show run, iter_count, inner_func_eval_count, phi_x_bar_new
    end
    
    if verbose
        @show(line_search_fail_count)
    end
    
    final_obj_vals[run] = final_obj_val
    line_search_fail_counts[run] = line_search_fail_count
    inner_func_eval_counts[run] = inner_func_eval_count
    outer_func_eval_counts[run] = outer_func_eval_count
    iter_counts[run] = iter_count
end

# Print table statistics
@printf("====Table Values====\n")
@printf("EnKSGD Mean Log10 Objective Value: %1.1E\n", mean(log.(10,final_obj_vals)))
@printf("EnKSGD Median Log10 Objective Value: %1.1E\n", median(log.(10,final_obj_vals)))
@printf("EnKSGD Minimum Log10 Objective Value: %1.1E\n", minimum(log.(10,final_obj_vals)))
@printf("EnKSGD Maximum Log10 Objective Value: %1.1E\n", maximum(log.(10,final_obj_vals)))
@printf("EnKSGD Variance Log10 Objective Value: %1.1E\n", var(log.(10,final_obj_vals)))
@printf("EnKSGD Mean Number of Iterations: %d\n", mean(iter_counts))

finalize(nls)

