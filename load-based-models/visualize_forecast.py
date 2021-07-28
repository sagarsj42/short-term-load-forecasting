import csv

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import feed_back as fb
import prep_data

class Forecaster(object):
    def __init__(self, run_network=False, replace_anomalies=False):
        self.set_params(run_network=run_network, replace_anomalies=replace_anomalies)
        self.save_forecast(anomaly_treated=replace_anomalies)
        self.intervals = {'day': 96, 'week': 7*96, 'month': 31*96, 'year': 366*96}

    def set_params(self, replace_anomalies, run_network=False):
        self.data, test = prep_data.DataLoader(replace_anomalies=replace_anomalies, correct_dst=True).get_data()
        self.data.extend(test)
        self.actual = list(map(lambda x : x[1][0][0], self.data))
        if run_network:
            net = fb.Network(hidden_layers=15, cache_len=15, replace_anomalies=replace_anomalies)
            net.SGD(mini_batch_size=10, eta=0.3, mu=0.6, epochs=None, eta_steps=10, monitor_session=False)
        filename = 'best_params.npz'
        with np.load(filename) as paramsfile:
            self.weights = paramsfile['weights']
            self.biases = paramsfile['biases']

    def save_forecast(self, anomaly_treated):
        self.timestamps = pd.read_csv('LD2011_2014_N.csv').YMDHMS.values[365*96 - 1:]
        self.forecast = [fb.feedforward(x[0], self.weights, self.biases)[0][0] for x in self.data]
        filename = 'LD2012_2014_NF'
        filename += 'A.csv' if anomaly_treated else '.csv'
        with open(filename, mode='w', newline='') as file:
            filewriter = csv.writer(file, delimiter=',', quotechar='|')
            filewriter.writerow(['YMDHMS', 'Actual', 'Forecast'])
            [filewriter.writerow([t, a, f]) for t, a, f in zip(self.timestamps, self.actual, self.forecast)]

    def plot_comparison(self, type='day', start='2013-01-01'):
        start += ' 00:00:00'
        index = np.where(self.timestamps == start)[0][0]
        last = index + self.intervals[type]
        actual = [a[1][0][0] for a in self.data[index:last]]
        forecast = self.forecast[index:last]
        self.plot_cmp(actual, forecast, type, start)

    def plot_cmp(self, actual, forecast, type, start):
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        ax.plot(actual, label='Actual Load')
        ax.plot(forecast, label='Forecasted Load')
        ax.set_title('Forecast type: ' + type + ', ' + start + ' onwards')
        ax.set_xlabel('Time Intervals')
        ax.set_ylabel('Normalized Load')
        plt.legend()
        plt.show()

    def plot_differences(self, start=0, end=1096*96):
        self.differences = list(map(lambda a1, a2 : (a1 - a2) / a1 * 100, self.actual, self.forecast))
        for d in self.differences:
            if abs(d) > 30:
                print(self.timestamps[self.differences.index(d)] + ' ' + str(d))
        fig = plt.figure(2)
        ax = fig.add_subplot(111)
        ax.plot(self.differences[start:end])
        ax.set_xlabel('Time Intervals')
        ax.set_ylabel('Forecast Differences')
        ax.set_title('Difference in the Actual & Predicted Load')
        plt.show()

if __name__ == "__main__":
    f = Forecaster(replace_anomalies=False)
    f.plot_comparison(type='week', start='2014-06-10')
    #f.plot_differences()