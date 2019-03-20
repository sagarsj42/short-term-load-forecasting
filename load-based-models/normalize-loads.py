import csv

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv("LD2011_2014.csv")
timestamps = data.YMDHMS
load = np.array(data.Load_kW.values)
load /= 500000
with open("LD2011_2014_N.csv", mode="w", newline='') as newfile:
    filewriter = csv.writer(newfile, delimiter=',', quotechar='|')
    filewriter.writerow(['YMDHMS', 'Load'])
    [filewriter.writerow([t, ln])for t, ln in zip(timestamps, load)]
newfile.close()

plt.plot(load)
plt.show()