import numpy as np
import matplotlib.pyplot as plt

table = np.loadtxt('Scan3D.txt', skiprows=14, dtype=float,
                   delimiter=';', usecols=(0, 1, 2, 3), encoding="ISO-8859-1")

table = np.transpose(table)

index_array = np.where(table[2] == 3.0)[0]

image = np.zeros((int(np.sqrt(len(index_array))), int(np.sqrt(len(index_array)))))

for i in range(len(index_array)):
    index = index_array[i]
    x_index = int(table[0][index] - 24.4)
    y_index = int(table[1][index] - 24.1)
    image[x_index][y_index] = table[3][index]

realx = [24.4, 25.4, 26.4, 27.4, 28.4, 29.4]
realy = [24.1, 25.1, 26.1, 27.1, 28.1, 29.1]
plt.imshow(image)

plt.xlabel("$x$ [mm]")
plt.ylabel("$y$ [mm]")
plt.gca().set_xticks(range(len(realx)))
plt.gca().set_yticks(range(len(realx)))
plt.gca().set_xticklabels(realx)
plt.gca().set_yticklabels(realy)
plt.colorbar()
plt.show()
