import pandas as pd
from numpy import inf

#loadfile = 'LD2011_2014.csv'
loadfile = 'TPC_Load.csv'
data = pd.read_csv(loadfile)
max = 0
min = inf
#for d in data.Load[365*96 - 1:]:
for d in data.Load:
    max = d if max < d else max
    min = d if min > d else min
print("Min: {0} \t Max: {1}.".format(min, max))
#Max value turns out to be 452,486 kW. So, normalizing factor = 500,000 kW.
#Max turns out to be 313,595 kWh. So, normalizing factor = 400,000 kWh.