import csv

import pandas as pd
import numpy as np
#from matplotlib import pyplot as plt

data = pd.read_csv("../../data/LD2011_2014.txt", sep=';')
(rows, columns) = data.shape
total_loads = np.zeros(rows)
time_stamps = []

for c, r in data.iteritems():
    if c == "YYYY-MM-DD HH:MM:SS":
        time_stamps = data[c].tolist()
    else:
        count = 0
        for x in data[c]:
            x_type = type(x)
            if not (x_type == type(np.int64(45)) or x_type == type(np.float64(45.45)) \
                    or x_type == type(45) or x_type == type(45.45)):
                t = x.split(',')
                x = int(t[0])
            total_loads[count] += int(x)
            count = count + 1

with open("LD2011_2014_total.csv", "w", newline='') as loadfile:
    filewriter = csv.writer(loadfile, delimiter=',', quotechar='|')
    filewriter.writerow(['YMDHMS', 'Load_kW'])
    for x, y in zip(time_stamps, total_loads):
        filewriter.writerow([x, y])