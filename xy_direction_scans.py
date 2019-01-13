import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


table_xscan = np.loadtxt("ScanXDirectionZ0_2.txt", skiprows=14, dtype=float, delimiter=';', usecols=(0, 3), encoding="ISO-8859-1")
table_yscan = np.loadtxt("ScanYDirectionZ0_2.txt", skiprows=14, dtype=float, delimiter=';', usecols=(1, 3), encoding="ISO-8859-1")
table_sim = np.loadtxt("simulation_cuts_xy.txt", skiprows =0, dtype=float)
table_sim_z = np.loadtxt("simulation_cuts_z.txt", skiprows =0, dtype=float)

x_values = np.transpose(table_xscan)[0]
x_intensities = np.transpose(table_xscan)[1]

y_values = np.transpose(table_yscan)[0]
y_intensities = np.transpose(table_yscan)[1]

sim_values = np.transpose(table_sim)[0]
x_sim_intensities = np.transpose(table_sim)[1]
y_sim_intensities = np.transpose(table_sim)[2]

sim_values_z = np.transpose(table_sim_z)[0]
z_sim_intensities = np.transpose(table_sim_z)[1]

simulation_x_interpol = interp1d(sim_values, x_sim_intensities, 'cubic')
simulation_y_interpol = interp1d(sim_values, y_sim_intensities, 'cubic')

def fit_function_x(x, x_shift, scale):
    if(hasattr(x, "__len__")):
        return [fit_function_x(x[i], x_shift, scale) for i in range(len(x))]
    if x-x_shift < min(sim_values):
        return simulation_x_interpol(min(sim_values))*scale
    if x-x_shift > max(sim_values):
        return simulation_x_interpol(max(sim_values))*scale
    return simulation_x_interpol(x-x_shift)*scale

def fit_function_y(x, x_shift, scale):
    if(hasattr(x, "__len__")):
        return [fit_function_y(x[i], x_shift, scale) for i in range(len(x))]
    if x-x_shift < min(sim_values):
        return simulation_y_interpol(min(sim_values))*scale
    if x-x_shift > max(sim_values):
        return simulation_y_interpol(max(sim_values))*scale
    return simulation_y_interpol(x-x_shift)*scale


initial_params_x = [27.1, 2.4]
params, cov = curve_fit(fit_function_x, x_values, x_intensities, p0 = initial_params_x)
print(params)

indices_filtered = list(filter(lambda i: abs(sim_values[i]) < 6, range(110)))
sim_values_filtered = [sim_values[i] for i in indices_filtered]
x_sim_intensities_filtered = [x_sim_intensities[i] for i in indices_filtered]

#print(len(sim_values_filtered))
#print(len(x_sim_intensities_filtered))
#plt.plot(sim_values_filtered, x_sim_intensities_filtered, 'x')

x_values_plot = np.linspace(min(x_values), max(x_values), num=500)
plt.plot(x_values, x_intensities, 'x')
plt.plot(x_values_plot, fit_function_x(x_values_plot, *initial_params_x))
plt.legend(["measured values", "FLUKA-Simulation"])
plt.xlabel("$x$ [mm]")
plt.ylabel("Intensity [a.u]")

plt.figure()

initial_params_y = [26.8, 2.4]
y_values_plot = np.linspace(min(y_values), max(y_values), num=500)
plt.plot(y_values, y_intensities, 'x')
plt.plot(y_values_plot, fit_function_y(y_values_plot, *initial_params_y))
plt.legend(["measured values", "FLUKA-Simulation"])
plt.xlabel("$y$ [mm]")
plt.ylabel("Intensity [a.u]")

plt.show()
