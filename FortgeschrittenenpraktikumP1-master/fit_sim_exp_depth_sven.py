import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

table1 = np.loadtxt('praktikum001_fort.21.lis', skiprows=9, dtype=float)

table2 = np.loadtxt('depthDose2.txt', skiprows=14, dtype=float,
                   delimiter=';', usecols=(2, 3), encoding="ISO-8859-1")

table2 = np.transpose(table2)

x_coord = np.zeros(len(table1[0]) * len(table1))
x_data = np.zeros(len(table1[0]) * len(table1))

x_coord2 = np.zeros(len(table2[0]))
x_data2 = np.zeros(len(table2[0]))

for i in range(len(table2[0])):
    x_coord2[i] = table2[0][i]
    x_data2[i] = table2[1][i]

for i in range(len(table1)):
    for j in range(len(table1[0])):
        x_data[10 * i + j] = table1[i][j] * 2000		# Where does that 2000 come from?
        x_coord[10 * i + j] = 1.35e-2 * (10 * i + j) - 1	# Replaced 1.35e-3 cm by 1.35e-2 mm, so we measure everything in mm

simulation_interpolated = interp1d(x_coord, x_data, kind='cubic')

def fit_function(x, x_shift, y_shift, scale):
	if(hasattr(x, "__len__")):
		return [fit_function(x[i], x_shift, y_shift, scale) for i in range(len(x))]
	if x-x_shift < min(x_coord):
		return simulation_interpolated(min(x_coord))*scale+y_shift
	if x-x_shift > max(x_coord):
		return simulation_interpolated(max(x_coord))*scale+y_shift
	return simulation_interpolated(x-x_shift)*scale+y_shift

initialParams = [-2, 0, 3]

params, cov = curve_fit(fit_function, x_coord2, x_data2, p0=initialParams)

x_coord_mod = [x-params[0] for x in x_coord2]
x_data_mod = [ (y-params[1])/params[2] for y in x_data2]


print(params)
print(cov)
shifted_x_coords = np.linspace(-3,7, num=500)
plt.plot(x_coord2,x_data2,'b+')
plt.plot(shifted_x_coords, [fit_function(x, *params) for x in shifted_x_coords], 'r-')
plt.xlabel("$z$ [mm]")
plt.ylabel("intensity [a.u]")
plt.legend(["measured data", "FLUKA simulation"])
plt.show()


