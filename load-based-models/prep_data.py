import pandas as pd
import numpy as np

class DataLoader(object):
    def __init__(self):
        self.data = pd.read_csv("LD2011_2014_N.csv")
        self.timestamps = self.data.YMDHMS.values
        self.load = np.array(self.data.Load_kW.values)
        self.lower = 365*96 - 1
        self.training_lim = 731*96
        self.upper = self.lower + 1096*96 + 1
        self.packets = []

    def get_weekday(self, year, month, date):
        day = 5 #Saturday on 1st Jan 2005
        day = day + year - 2005 + int(np.floor((year - 2005)) / 4)
        month_dict = {1: 0, 2: 3, 3: 3, 4: 6, 5: 8, 6: 11, 7: 13, 8: 16, 9: 19, 10: 21, 11: 24, 12: 26}
        day += month_dict[month]
        day += 1 if (year % 4 == 0 and month > 2) else 0
        day = (day + date - 1) % 7
        return day

    def int2bin(self, arr, val):
        ind = 1
        while val > 0:
            arr[-ind] = val % 2
            val = np.floor(val / 2)
            ind += 1
        return arr

    def get_MDHS(self, month, day, hour, subhour):
        m = np.zeros(4)
        d = np.zeros(3)
        h = np.zeros(5)
        s = np.zeros(2)
        return (self.int2bin(m, month), self.int2bin(d, day), self.int2bin(h, hour), self.int2bin(s, subhour))

    def get_calendar_params(self, timestamp):
        year = int(timestamp[0:4])
        month = int(timestamp[5:7])
        date = int(timestamp[8:10])
        day = self.get_weekday(year, month, date)
        hour = int(timestamp[11:13])
        subhour = int(timestamp[14:16])/15
        return self.get_MDHS(month, day, hour, subhour)

    def get_data(self):
        for i in range(self.lower, self.upper):
            prevh1 = self.load[i - 4]
            prevh2 = self.load[i - 8]
            prevh3 = self.load[i - 12]
            prevd = self.load[i - 96]
            prevw = self.load[i - 96*7]
            month, day, hour, subhour = self.get_calendar_params(self.timestamps[i])
            output = np.array(self.load[i]).reshape(1, 1)
            input_list = [prevh1, prevh2, prevh3, prevd, prevw]
            [input_list.append(m) for m in month]
            [input_list.append(d) for d in day]
            [input_list.append(h) for h in hour]
            [input_list.append(s) for s in subhour]
            len_ip = len(input_list)
            input_list = np.array(input_list).reshape(len_ip, 1)
            self.packets.append((input_list, output))
        return (self.packets[: self.training_lim], self.packets[self.training_lim:])

if __name__ == "__main__":
    prep = DataLoader()
    training_data, test_data = prep.get_data()
    print("Some training data:")
    [print(a) for a in training_data[-10:]]
    print("\nSome test data:")
    [print(a) for a in test_data[-10:]]