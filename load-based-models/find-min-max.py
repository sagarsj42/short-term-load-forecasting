import pandas as pd
from numpy import inf

data = pd.read_csv("LD2011_2014.csv")
max = 0
min = inf
for d in data.Load_kW[365*96 - 1:]:
    max = d if max < d else max
    min = d if min > d else min
print("Min: {0} \t Max: {1}.".format(min, max))
#Max value turns out to be 452,486. So, normalizing factor = 400, 000.