import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("LD2011_2014_total.csv")
lower = 365*96 - 1
upper = 731*96 - 1
plt.plot(data.Load_kW[lower:])
plt.xlabel('Time')
plt.ylabel('Load (kW)')
#plt.title('2011 to 2014 Total Load Data: Equalised')
plt.show()