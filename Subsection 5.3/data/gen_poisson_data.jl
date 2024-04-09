using Distributions
using JLD
using LinearAlgebra

Sigma = diagm(10.0 .^ (range(-5,-1,step=0.1)))
d = size(Sigma, 1)

F = cholesky(Sigma)
num_samples = 189
X = (F.L*randn(d, num_samples))'

w = randn(d)
lambdas = exp.(X*w)

S = zeros(num_samples)
for i in 1:num_samples
  p = Poisson(lambdas[i])
  S[i] = rand(p)
end

@save "poisson_data.jld" X w S
