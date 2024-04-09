using LinearAlgebra
using NLPModels
using NLSProblems

# Define a "CompositeObjFun" type for evaluating a composite objective function and
# its derivatives
mutable struct CompositeObjFun
    func_eval # Function that evaluates the composite objective function
    grad_eval # Function that evaluates the gradient of the composite objective function
    hess_eval # Function that evaluates the Hessian of the composite objective function
    inner_func_eval # Function that evaluates the inner part of the composite objective function
    inner_func_jacob # Function that evaluates the Jacobian of the inner function
    outer_func_eval # Function that evaluates the outer part of the composite objective function
    outer_func_grad # Function that evaluates the gradient of the outer function
    outer_func_hess # Function that evaluates the hessian of the outer function
end

# Function to evaluate a linear least squares problem
function linear_ls(;A=[5 1; 1 3],b=[1.4;2.0])
    
    func_eval(x) = (1/2)*(A*x - b)'*(A*x - b)
    grad_eval(x) = A'*(A*x - b)
    hess_eval(x) = A'*A
    inner_func_eval(x) = A*x - b
    inner_func_jacob(x) = A
    outer_func_eval(x) = (1/2)*x'*x
    outer_func_grad(x) = x
    outer_func_hess(x) = I
    
    return CompositeObjFun(func_eval,
                           grad_eval,
                           hess_eval,
                           inner_func_eval,
                           inner_func_jacob,
                           outer_func_eval,
                           outer_func_grad,
                           outer_func_hess)
end

# Function to evaluate a linear least squares problem
# with noise added to the residual function
function linear_ls_additive_noise(res_noise;A=[5 1; 1 3],b=[1.4;2.0])
    
    inner_func_eval(x) = A*x - b + res_noise(x)
    inner_func_jacob(x) = A
    outer_func_eval(x) = (1/2)*x'*x
    outer_func_grad(x) = x
    outer_func_hess(x) = I
    func_eval(x) = outer_func_eval(inner_func_eval(x))
    grad_eval(x) = A'*inner_func_eval(x)
    hess_eval(x) = A'*A
    
    return CompositeObjFun(func_eval,
                           grad_eval,
                           hess_eval,
                           inner_func_eval,
                           inner_func_jacob,
                           outer_func_eval,
                           outer_func_grad,
                           outer_func_hess)
end

# Function to evaluate an NLSProblem from the NLPModels.jl API
function nls_objective(nls; obj_weight=1.0)
    func_eval(x) = obj(nls, x)
    grad_eval(x) = grad(nls, x)    
    L(x) = hess(nls, x; obj_weight=obj_weight)
    function hess_eval(x)
        B = L(x)
        return Array(B) + Array(B') - Diagonal(diag(B))
    end
    inner_func_eval(x) = residual(nls, x)
    inner_func_jacob(x) = Array(jac_residual(nls, x))
    outer_func_eval(x) = (1/2)*x'*x
    outer_func_grad(x) = x
    outer_func_hess(x) = I
    
    return CompositeObjFun(func_eval,
                           grad_eval,
                           hess_eval,
                           inner_func_eval,
                           inner_func_jacob,
                           outer_func_eval,
                           outer_func_grad,
                           outer_func_hess)
end

