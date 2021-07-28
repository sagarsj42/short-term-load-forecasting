import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv("../../data/LD2012.txt", sep=';')
(rows, columns) = data.shape
total_load = np.zeros(rows)
lower = 9600
upper = 10272
offset = 0
load_vals = np.zeros(upper - lower + 1)

for x in data.MT_100[lower:upper]:
    x_type = type(x)
    if x_type == type(np.int64(45)) or x_type == type(np.float64(45.45))  \
            or x == type(45) or x == type(45.45):
        load_vals[offset] = x
    else:
        t = x.split(',')
        x = int(t[0])
        load_vals[offset] = x
    offset = offset + 1

plt.plot(load_vals)
plt.show()