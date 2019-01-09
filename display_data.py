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

plt.plot(x_coord, x_data)
plt.plot(x_coord, decay(x_coord, popt[0], popt[1], popt[2]))
plt.show()
