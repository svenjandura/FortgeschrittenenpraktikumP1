import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
        x_data[10 * i + j] = table1[i][j] * 2000
        x_coord[10 * i + j] = 1.35e-3 * (10 * i + j)

x_diff = np.zeros(5)

for i in range(5):
    x_diff[i] = x_data2[i] - x_data[np.where(np.floor(x_coord*10)/10
                                        == x_coord2[i])[0][0]]

plt.plot(x_coord, x_data)
plt.figure()
plt.plot(x_coord2, x_data2)
plt.show()

def difference(x, num, dev):
    return x - num * x_data[np.where(np.floor(x_coord*10)/10 ==
                                     x_coord2[np.where(x_data2 == x)[0][0]])[0][0]]

for i in range(19):
    try:
        print(i)
        popt, pcov = curve_fit(difference, x_data2[0:3], np.zeros(3), bounds=(0, [1e16, 0.01]))
        break
    except:
        pass

plt.plot(x_coord, popt[0] * x_data)
plt.plot(x_coord2 - popt[1], x_data2)
plt.show()
