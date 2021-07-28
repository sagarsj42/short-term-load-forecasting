import os

import pandas as pd
from matplotlib import pyplot as plt

os.chdir('../../data/UI_Bills')
#data = pd.read_csv("LD2011_2014_total.csv")
data = pd.read_csv('SLDC_Load.csv')
lower = 365*96 - 1
upper = 731*96 - 1
#plt.plot(data.Load_kW[lower:])
plt.plot(data.Load_kWh[:])
plt.xlabel('Time')
plt.ylabel('Load (kW)')
#plt.title('2011 to 2014 Total Load Data: Equalised')
plt.show()