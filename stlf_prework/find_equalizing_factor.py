'''A general factor by which the data for 2011 lags behind that of 2012.'''

import csv

import pandas as pd
import numpy as np

data_all = pd.read_csv("LD2011_2014_total_equalized.csv")
data_2012 = pd.read_csv("LD2012_total.csv")
r, c = data_all.shape
factors = np.zeros(r)
div = data_2012["Load (kW)"].tolist()
timestamps = data_all["YYYY-MM-DD HH:MM:SS"].tolist()

count1, count2 = 0, 0
for x in data_all["Load_kW"]:
    factors[count1] = div[count2] / x #Multiply by this factor while equalizing.
    count1 = count1 + 1
    count2 = count2 + 1
    if count1 == 365*96 - 1 or count1 == 731*96 - 1 or count1 == 1096*96 - 1: count2 = 0

with open("equalizing_factors_new_2011_2014.csv", 'w', newline='') as eq_file:
    filewriter = csv.writer(eq_file, delimiter=',', quotechar='|')
    filewriter.writerow(["YYYY-MM-DD HH:MM:SS", "Factor"])
    for x, y in zip(timestamps, factors):
        filewriter.writerow([x, y])
    filewriter.writerow(["--------------------------", "--------------------------"])
    filewriter.writerow(["Average 1", sum(factors[0 : 365*96-1]) / (365*96 - 1)])
    filewriter.writerow(["Average 2", sum(factors[365*96-1 : 731*96-1]) / (366*96)])
    filewriter.writerow(["Average 3", sum(factors[731*96-1 : 1096*96-1]) / (365*96)])
    filewriter.writerow(["Average 4", sum(factors[1096*96-1 : 1461*96-1]) / (365*96)])