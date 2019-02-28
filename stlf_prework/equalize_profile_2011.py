import csv
import pandas as pd

data_uneq = pd.read_csv("LD2011_2014_total.csv")
factors = pd.read_csv("equalizing_factors_2011_2014.csv")
timestamp = data_uneq["YMDHMS"].tolist()
load = data_uneq["Load_kW"].tolist()

with open("LD2011_2014_total_equalized.csv", "w", newline='') as eq_file:
    fwriter = csv.writer(eq_file, delimiter=',', quotechar='|')
    fwriter.writerow(["YMDHMS", "Load_kW"])
    for i in range(0, 365*96-1):
        fwriter.writerow([timestamp[i], load[i] * 1.7])
    for i in range(365*96-1, 140256):
        fwriter.writerow([timestamp[i], load[i]])