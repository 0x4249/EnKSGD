using JLD
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

# Setup gain function
function gain(x)
    return 100*tanh(x/25)
end

# Plotting setup
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["text.usetex"] = true
rcParams["figure.autolayout"] = true
color_map = get_cmap("twilight_shifted")

# Input signal figure
figure()
axes_input_signal = plt.gca()
axes_input_signal.set_xlim(x[1],x[end])
axes_input_signal.set_ylim(-45,45)
axes_input_signal.grid(true)
axes_input_signal.tick_params(axis="x", labelsize=26)
axes_input_signal.tick_params(axis="y", labelsize=26)
xlabel(L"t", fontsize=26)
ylabel(L"\mathbf{x}(t)", fontsize=26)
title("Input Signals", fontsize=26)
axes_input_signal.plot(x,y_rect, color="b", label="True Signal")
axes_input_signal.legend(fontsize=26,loc="lower right")
axes_input_signal.text(-0.3, 1.1, "(A)", size=20, weight="bold", transform=axes_input_signal.transAxes)

# Amplified signal figure
figure()
axes_amplified_signal = plt.gca()
axes_amplified_signal.set_xlim(x[1],x[end])
axes_amplified_signal.set_ylim(-120,120)
axes_amplified_signal.grid(true)
axes_amplified_signal.tick_params(axis="x", labelsize=26)
axes_amplified_signal.tick_params(axis="y", labelsize=26)
xlabel(L"t", fontsize=26)
ylabel(L"\mathcal{G}(\mathbf{x}(t))", fontsize=26)
title("Amplified Signals", fontsize=26)
axes_amplified_signal.plot(x,y_obs, linestyle="-.", color="r", label="Measured Signal")
axes_amplified_signal.plot(x,gain.(y_rect), color="b", label="True Signal")
axes_amplified_signal.legend(fontsize=26,loc="lower right")
axes_amplified_signal.text(-0.37, 1.1, "(B)", size=20, weight="bold", transform=axes_amplified_signal.transAxes)

