import numpy as np
import matplotlib.pyplot as plt

table = np.loadtxt("praktikum001_fort.22.lis", skiprows = 9, dtype = float)

def get_table_entry(x,y,z):
	index = z + 1000*x + 1000*110*y
	return table[int(index/10)][index%10]

Z0 = 1.5
z_index = int((Z0+1)/13.5*1000)

middle_index = 55

x_values = [-25 + i*1.0/110*50 for i in range(110)]
x_intensities = [get_table_entry(x, middle_index, z_index) for x in range(110)]
y_intensities = [get_table_entry(middle_index, y, z_index) for y in range(110)]

z_values = [-1 + i*1.0/1000*13.5 for i in range(z_index)]
z_intensities = [get_table_entry(middle_index, middle_index, z) for z in range(z_index)]

np.savetxt("simulation_cuts_xy.txt", np.array([x_values, x_intensities, y_intensities]).transpose())
np.savetxt("simulation_cuts_z.txt", np.array([z_values, z_intensities]).transpose())
