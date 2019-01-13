import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

table = np.loadtxt('depthDose2.txt', skiprows=14, dtype=float,
                   delimiter=';', usecols=(0, 1, 2, 3), encoding="ISO-8859-1")

table = np.transpose(table)

def decay (x, a, b, c):
    return a * (x - b)**(-2) + c

x_inf = np.amin(table[0])
x_sup = np.amax(table[0])
y_inf = np.amin(table[1])
y_sup = np.amax(table[1])
z_inf = np.amin(table[2])
z_sup = np.amax(table[2])

x_coord = np.zeros(len(table[0]))
x_data = np.zeros(len(table[0]))

for i in range(len(table[0])):
    x_coord[i] = table[2][i]
    x_data[i] = table[3][i]

a = 0.3017 * (0.07467525 + 0.195 * np.sqrt(0.065169))/0.32592681
b = (-0.0975 - np.sqrt(0.065169))/0.5709

popt, pcov = curve_fit(decay, x_coord, x_data, p0=(0.56, -1.366, 0))
#popt, pcov = curve_fit(decay, x_coord, x_data, p0=(a, b))

yerror_relative = [0.005]*12+[0.01]*2+[0.01*x_coord[i]/x_coord[13] for i in range(14, len(x_coord))]
yerror = [x_data[i]*yerror_relative[i] for i  in range(len(x_data))]
xerror = [0.05]*len(x_data)

print(yerror_relative)
print(popt)
plt.errorbar(x_coord, x_data, yerr=yerror, xerr=xerror, fmt='x')
plt.plot(x_coord, decay(x_coord, popt[0], popt[1], popt[2]))
plt.xlabel("$z$ [mm]")
plt.ylabel("intensity [a.u]")
plt.legend(['fit', 'measured data'])
plt.show()
