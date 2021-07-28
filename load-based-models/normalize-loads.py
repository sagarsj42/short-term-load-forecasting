import csv

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#loadfile = 'LD2011_2014.csv'
loadfile = 'TPC_Load.csv'
data = pd.read_csv(loadfile)
timestamps = data.YMDHMS
load = np.array(data.Load.values)
#load /= 500000
load /= 400000
#load = (load - 282) / (452486 - 282)

newfile = loadfile[:-4] + '_N.csv'
with open(newfile, mode="w", newline='') as file:
    filewriter = csv.writer(file, delimiter=',', quotechar='|')
    filewriter.writerow(['YMDHMS', 'Load'])
    [filewriter.writerow([t, ln])for t, ln in zip(timestamps, load)]
file.close()

plt.plot(load)
plt.title('Normalized Data')
plt.xlabel('Time Instant')
plt.ylabel('Normalized Load Value')
plt.show()