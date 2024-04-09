using JLD

# Setup sinusoidal function parameters
omega = 6*pi
amp = 2e1

# Setup grid
start = 0
stop = 1
num_samples = 101
x = range(start,stop,length=num_samples)

# Calculate sinusoidal function
y = amp*sin.(omega*x)

# Rectify sinusoidal function
y_rect = deepcopy(y)
y_rect[y_rect .<= 0] .= 0

# Apply gain function and add Gaussian noise 
sigma_n = 15
gain(x) = 100*tanh(x/25)
y_out = gain.(y_rect) + sigma_n*randn(num_samples)

@save "sinusoidal_data.jld" omega amp sigma_n y_out y_rect y x
