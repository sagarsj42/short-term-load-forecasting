import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv("LD2012_01_07.txt", sep=";")
(rows, columns) = data.shape
total_load = np.zeros(rows)

for key, value in data.iteritems():
    loads_temp = np.zeros(rows)
    if key == "YYYY-MM-DD HH:MM:SS":
        continue
    else:
         i = 0
         for load_val1 in data[key].values:
             x = type(load_val1)
             if x == type(np.int64(45)) or x == type(np.float64(45.45)) \
                     or x == type(34) or x == type(34.34):
                 loads_temp[i] = load_val1
             else:
                 t = load_val1.split(',')
                 loads_temp[i] = int(t[0])
             total_load += loads_temp
             i = i + 1
plt.plot(total_load)
plt.show()