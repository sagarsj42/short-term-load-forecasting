import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("countries.csv")
us = data[data.country == "United States"]
china = data[data.country == "China"]
india = data[data.country == "India"]
plt.plot(us.year, us.population / us.population.iloc[0])
plt.plot(china.year, china.population / china.population.iloc[0])
plt.plot(india.year, india.population / india.population.iloc[0])
plt.legend(['US', 'China', 'India'])
plt.show()