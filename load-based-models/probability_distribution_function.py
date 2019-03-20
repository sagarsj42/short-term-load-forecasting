import numpy as np
import pandas as pd
import scipy.stats as sts
from matplotlib import pyplot as plt

class PDF(object):
    def __init__(self, confidence_level=2.0, cutoff=20):
        self.set_data()
        self.form_intervals()
        self.fit_rows()
        self.confidence_lev = confidence_level
        self.set_limits()
        self.cutoff = cutoff
        self.anomalies = []

    def set_data(self):
        self.data = pd.read_csv("LD2011_2014_N.csv")
        self.lower_lim = 365*96 - 1
        self.upper_lim = self.lower_lim + 1096*96 + 1
        self.days = int((self.upper_lim - self.lower_lim) / 96)

    def form_intervals(self):
        self.intervals = np.zeros((96, self.days))
        for d in range(0, self.days):
            for i in range(0, 96):
                self.intervals[i][d] = self.data.Load[self.lower_lim + d*96 + i]

    def fit_rows(self):
        self.params = []
        for i in range(0, 96):
             self.params.append(sts.norm.fit(self.intervals[i]))

    def set_limits(self):
        self.limits = [(m - self.confidence_lev*d, m + self.confidence_lev*d) for m, d in self.params]

    def detect_anomalies(self):
       for d in range(0, self.days):
           spikes = 0
           diff = 0
           for i in range(0, 96):
               low, high = self.limits[i]
               val = self.intervals[i][d]
               if val < low or val > high:
                   spikes += 1
                   diff += low - val if val < low else val - high
           if spikes > self.cutoff:
                self.anomalies.append((self.data.YMDHMS[self.lower_lim + d*96][:10], spikes, diff))

    def print_anomalies(self):
       [print("Date: {0} \t Spikes: {1} \t Diff.: {2}".format(a, b, c)) for a, b, c in self.anomalies]
       print("{0} anomalies detected.".format(len(self.anomalies)))

    def plot_limits(self, date=None):
        fig, ax = plt.subplots()
        x = np.arange(0, 24, 0.25)
        lower = []
        higher = []
        mean = []
        [mean.append(m) for m, d in self.params]
        for l, h in self.limits:
            lower.append(l)
            higher.append(h)
        l1, = ax.plot(x, lower, 'c:')
        l2, = ax.plot(x, higher, 'g:')
        l3, = ax.plot(x, mean, 'b')
        if date:
            load = self.extract_load(date)
            l4, = ax.plot(x, load, 'r--')
            ax.legend((l1, l2, l3, l4), ('Lower', 'Higher', 'Mean', 'Load on ' + date), loc='upper left', shadow=True)
            ax.set_title('Probability Distribution Fitted Limits for Daily Load')
            ax.set_xlabel('Time (hrs.) -->')
            ax.set_ylabel('Load (kW) -->')
            plt.show()
            return
        ax.legend((l1, l2, l3), ('Lower', "Higher", 'Mean'), loc='upper left', shadow=True)
        ax.set_title('Probability Distribution Fitted Limits for Daily Load')
        ax.set_xlabel('Time (hrs.) -->')
        ax.set_ylabel('Load (kW) -->')
        plt.show()

    def extract_load(self, date):
        date += " 00:00:00"
        p = pd.Index(self.data.YMDHMS).get_loc(date)
        return self.data.Load[p: p+96]

    def get_anomalies(self):
        baddates = [d for d, s, diff in self.anomalies]
        return baddates

if __name__ == "__main__":
    pdf = PDF(2.0, 15)
    pdf.detect_anomalies()
    pdf.print_anomalies()
    pdf.plot_limits(date="2012-12-25")