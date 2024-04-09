using JLD
using LinearAlgebra
using Printf
using PyPlot

# Load data
data = load("data/sinusoidal_data.jld")
omega = data["omega"]
amp = data["amp"]
sigma_y = data["sigma_n"]
y_obs = data["y_out"]
y_rect = data["y_rect"]
y = data["y"]
x = data["x"]
d = length(y_rect)
m = d
@show m, d

W_sqrt = I

# Setup gain function
function gain(x)
    return 100*tanh(x/25)
end

# Aggregate parameters
total_runs = 30
maxIter = 61
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
    x_0 = zeros(d)
    K = 101
    sigma = 1e-2
    delta = 1e-3
    beta = 1e-6
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
    
    # State space regularization function setup
    lambda_x = 1e10
    B = zeros(d,d); B[1,1] = 1; B[end,end] = 1;
    
    # Observation space regularization function setup
    lambda_y = 5e0
    D = diagm(0 => ones(d), 1 => -ones(d-1))
    D = D[1:end-1,:]
    
    # Evaluate forward map (likelihood) at mean of particle locations
    G_x_bar = gain.(x_bar)
    inner_func_eval_count += 1
    
    # Evaluate loss function    
    D_x_bar = (1/2)*norm(W_sqrt*(G_x_bar - y_obs),2)^2
    outer_func_eval_count += 1
    
    # Evaluate objective function
    phi_x_bar = D_x_bar + (lambda_y/2)*norm(D*G_x_bar,2)^2 + (lambda_x/2)*(x_bar'*B*x_bar)
    
    # Store evaluation
    x_vals[1,:] = x_bar
    phi_vals[1] = phi_x_bar
    
    # EnKF Loop
    for i in 1:maxIter
        
        # Evaluate forward map at particle locations
        G = zeros(m,K)
        for k in 1:K
            G[:,k] = gain.(X[:,k])
            inner_func_eval_count += 1
        end
        
        # Calculate forward map deviations
        G_devs = G*Pi_K
        if verbose
            @show norm(G_devs)
        end
        
        # Evaluate loss function gradient
        grad_loss_x_bar = W_sqrt'*W_sqrt*(G_x_bar - y_obs)
        outer_func_grad_eval_count += 1
        
        # Evaluate loss function Hessian
        H_loss_x_bar = W_sqrt'*W_sqrt
        outer_func_hess_eval_count += 1
        if verbose
            @show norm(H_loss_x_bar)
        end
        
        # Compute objective function gradient projected onto particles
        q_loss = G_devs'*grad_loss_x_bar
        q_observation_space_reg = G_devs'*(lambda_y)*D'*D*G_x_bar
        q_state_space_reg = Y'*(lambda_x/2)*(B + B')*x_bar
        q = q_loss + q_observation_space_reg + q_state_space_reg
        
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
                
                G_x_bar_new = gain.(x_bar_new)
                inner_func_eval_count += 1
                
                D_x_bar_new = (1/2)*norm(W_sqrt*(G_x_bar_new - y_obs),2)^2
                outer_func_eval_count += 1
                
                phi_x_bar_new = D_x_bar_new + (lambda_y/2)*norm(D*G_x_bar_new,2)^2
                phi_x_bar_new += (lambda_x/2)*(x_bar_new'*B*x_bar_new)
                
                break
            end
            
            # Compute inverse of proposed transform matrix
            scaled_dt_prop = dt_prop/delta
            A = G_devs'*H_loss_x_bar*G_devs + (lambda_y)*G_devs'*D'*D*G_devs
            A += (lambda_x/2)*Y'*(B+B')*Y
            T_inv = I + (scaled_dt_prop/K)*(A)
            if verbose
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
            G_x_bar_prop = gain.(x_bar_prop)
            inner_func_eval_count += 1
            
            # Evaluate loss function
            D_x_bar_prop = (1/2)*norm(W_sqrt*(G_x_bar_prop - y_obs),2)^2
            outer_func_eval_count += 1
            
            # Evaluate objective function
            phi_x_bar_prop = D_x_bar_prop + (lambda_y/2)*norm(D*G_x_bar_prop,2)^2
            phi_x_bar_prop += (lambda_x/2)*(x_bar_prop'*B*x_bar_prop)
            
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
        Y_new = Y*T_up_sqrt + sqrt(beta*delta*dt)*randn(d,K)
        
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

# Reconstructed input signal figure
figure()
axes_reconstructed_input_signal = plt.gca()
axes_reconstructed_input_signal.set_xlim(x[1],x[end])
axes_reconstructed_input_signal.set_ylim(-45,45)
axes_reconstructed_input_signal.grid(true)
axes_reconstructed_input_signal.tick_params(axis="x", labelsize=26)
axes_reconstructed_input_signal.tick_params(axis="y", labelsize=26)
axes_reconstructed_input_signal.set_xlabel(L"t", fontsize=26)
axes_reconstructed_input_signal.set_ylabel(L"\mathbf{x}(t)", fontsize=26)
axes_reconstructed_input_signal.set_title("EnKF Reconstructed \n Input Signals", fontsize=26)
for run in 1:total_runs
    axes_reconstructed_input_signal.plot(x,all_x_vals[end,:,run], alpha=0.8, color=color_map(run/total_runs), label=("Reconstruction $(run)"))
end
axes_reconstructed_input_signal.text(-0.3, 1.1, "(D)", size=20, weight="bold", transform=axes_reconstructed_input_signal.transAxes)

