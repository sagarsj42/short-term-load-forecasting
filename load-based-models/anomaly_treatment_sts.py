import csv

import numpy as np
import pandas as pd

import vector_norm as vn
import probability_distribution_function as pdf

class AnomalyTreaterSts(object):
    def __init__(self):
        vno = vn.VectorNorm(confd_level=1.5, norm=2)
        vno.find_anomalies()
        vn_an = vno.get_anomalies()
        pdfo = pdf.PDF(confidence_level=2.0, cutoff=15)
        pdfo.detect_anomalies()
        pdf_an = pdfo.get_anomalies()
        self.anomalies = []
        self.common_anomalies = []
        self.consolidate_anomalies(vn_an, pdf_an)

    def consolidate_anomalies(self, vn_an, pdf_an):
        for v in vn_an:
            if v not in pdf_an:
                self.anomalies.append(v)
            else:
                self.common_anomalies.append(v)
        for p in pdf_an:
            self.anomalies.append(p)
        self.anomalies.sort()

    def print_anomalies(self):
        [print(a) for a in self.anomalies]
        print("{0} anomalies consolidated.".format(len(self.anomalies)))
        [print(a) for a in self.common_anomalies]
        print("{0} common bad dates found.".format(len(self.common_anomalies)))

    def replace_anomalies(self, use_OR=False):
        anomalies = self.anomalies if use_OR else self.common_anomalies
        data = pd.read_csv('LD2011_2014_N.csv')
        self.timestamps = data.YMDHMS.values
        self.load = data.Load.values
        for an in anomalies:
            an += ' 00:00:00'
            i = np.where(self.timestamps == an)[0][0]
            if i < 365*96 + 96*7:
                ic = i + 96*7
                while self.timestamps[ic][:10] in self.anomalies:
                    ic += 96*7
            else:
                ic = i - 96*7
                while self.timestamps[ic][:10] in self.anomalies:
                    ic -= 96*7
            self.load[i: i+96] = self.load[ic: ic+96]
        self.save_rectified_data()

    def save_rectified_data(self):
        with open('LD2011_2014_NA.csv', mode='w', newline='') as file:
            filewriter = csv.writer(file, delimiter=',', quotechar='|')
            filewriter.writerow(['YMDHMS', 'Load'])
            [filewriter.writerow([t, ld]) for t, ld in zip(self.timestamps, self.load)]

    def get_AND_anomalies(self):
        return self.common_anomalies

    def get_OR_anomalies(self):
        return self.anomalies

if __name__ == "__main__":
    ats = AnomalyTreaterSts()
    ats.replace_anomalies(use_OR=True)
    ats.print_anomalies()