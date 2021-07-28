import pandas as pd
from matplotlib import pyplot as plt

filename = 'LD2011_2014_N.csv'
data = pd.read_csv(filename)
load = data.Load.values
data.YMDHMS = pd.Series([d[:10] for d in data.YMDHMS.values])

def compare_profiles(day1, day2):
    load1 = data[data.YMDHMS == day1]
    load2 = data[data.YMDHMS == day2]
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.plot(range(load1.shape[0]), load1.Load, label=day1)
    ax.plot(range(load2.shape[0]), load2.Load, label=day2)
    ax.set_title('Comparison of Load Profiles')
    ax.set_xlabel('Time of Day')
    ax.set_ylabel('Load')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    compare_profiles('2013-12-25', '2013-12-26')