# The Ensemble Kalman-Stein Gradient Descent (EnKSGD) Paper
This folder contains the Julia code for running the numerical experiments discussed in the paper 
"EnKSGD: A Class Of Preconditioned Black Box Optimization And Inversion Algorithms"
by Brian Irwin and Sebastian Reich.


# Running The Code
## Subsection 5.1: Ill-Conditioned Linear Least Squares
To run the ill-conditioned linear least squares experiments from subsection 5.1, first open a Julia
REPL in the directory "Subsection 5.1". Then execute the following commands in the Julia REPL:
```
include("aggregate_enksgd_optimize_least_squares.jl")
include("aggregate_cfd_bfgs_optimize_least_squares.jl")
include("aggregate_enkf_optimize_least_squares.jl")
include("aggregate_cfd_gd_optimize_least_squares.jl")
include("aggregate_enksgd_optimize_noisy_least_squares.jl")
include("aggregate_cfd_bfgs_optimize_noisy_least_squares.jl")
include("aggregate_enkf_optimize_noisy_least_squares.jl")
include("aggregate_cfd_gd_optimize_noisy_least_squares.jl")
```
The first 4 of the 8 commands above produce Figure 1 (A) - (D) and the second 4 commands produce Figure 2 (E) - (H).


## Subsection 5.2: Nonlinear Least Squares
To run the nonlinear least squares experiments from subsection 5.2, first open a Julia REPL in the directory 
"Subsection 5.2". Then execute the following commands in the Julia REPL:
```
include("aggregate_enksgd_optimize_NLS_table.jl")
include("aggregate_enkf_optimize_NLS_table.jl")
```
Change the problem by changing the uncommented line in lines 8 - 18 of each file. Table 1 is produced by running
the 2 commands above for all 11 problems and recording the results. 


## Subsection 5.3: Poisson Regression
To run the Poisson regression experiments from subsection 5.3, first open a Julia REPL in the directory 
"Subsection 5.3". Then execute the following commands in the Julia REPL:
```
include("aggregate_enksgd_poisson_regression.jl")
include("aggregate_enkf_poisson_regression.jl")
```
The 2 commands above produce Figure 3 (A) - (B).


## Subsection 5.4: Signal Reconstruction
To run the signal reconstruction experiments from subsection 5.4, first open a Julia REPL in the directory 
"Subsection 5.4". Then execute the following commands in the Julia REPL:
```
include("visualize_signals.jl")
include("aggregate_enksgd_signal_reconstruction.jl")
include("aggregate_enkf_signal_reconstruction.jl")
```
The 3 commands above produce Figure 4 (A) - (D).


The numerical experiments code contained in this folder was originally tested using Julia 1.5.4 on a computer with the 
Ubuntu 20.04 LTS operating system installed.


