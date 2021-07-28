import os
import xml.etree.ElementTree as et
import csv

import xlrd

class SLDC_Data_Extractor(object):
    def __init__(self):
        os.chdir('../../data/UI_Bills')
        self.years = ('2012', '2013', '2014', '2015', '2016')
        self.months = {'Jan': (1, 31), 'Feb': (2, 28), 'Mar': (3, 31), 'Apr': (4, 30), 'May': (5, 31), 'June': (6, 30),
                  'July': (7, 31), 'Aug': (8, 31), 'Sept': (9, 30), 'Oct': (10, 31), 'Nov': (11, 30), 'Dec': (12, 31)}
        self.timestamped_load = []

    def get_weekend(self, weekstart, month, year):
        weekend = weekstart + 6
        return self.get_date(weekend, month, year)

    def get_date(self, date, month, year):
        if month == 'Feb' and int(year) % 4 == 0 and date > 29:
            date = date % 29
        elif (month == 'Feb' and int(year) % 4 == 0 and date == 29) or date == self.months[month][1]:
            return date
        elif date > self.months[month][1]:
            date %= self.months[month][1]
        return date

    def is_date_in_month(self, date, month, year):
        if month == 'Feb' and int(year) % 4 == 0 and date <= 29:
            return True
        elif date <= self.months[month][1]:
            return True
        else:
            return False

    def num2str(self, num):
        string = '0' + str(num) if num < 10 else str(num)
        return string

    def isXML(self, filename):
        if open(filename, errors='ignore').readline() == '<?xml version="1.0"?>\n':
            return True
        else:
            return False

    def extract_load_from_XML(self, filename):
        day_load = [0*i for i in range(96)]
        workbook = et.parse(filename)
        root = workbook.getroot()
        start = 3
        if list(root)[1].tag == '{urn:schemas-microsoft-com:office:office}OfficeDocumentSettings':
            start += 1
        worksheets = list(root)[start: start + 96]
        for c, ws in enumerate(worksheets):
            datacells = worksheets[c][0][16]
            load = datacells[3][0].text
            day_load[c] = load
        return day_load

    def extract_load_from_Excel(self, filename):
        day_load = [0*i for i in range(96)]
        xl_workbook = xlrd.open_workbook(filename)
        for index in range(96):
            xl_sheet = xl_workbook.sheet_by_index(index)
            day_load[index] = float(xl_sheet.cell(9, 5).value)
        return day_load

    def extract_load(self, filename):
        if self.isXML(filename):
            return self.extract_load_from_XML(filename)
        else:
            return self.extract_load_from_Excel(filename)

    def generate_timestamped_load(self, day_load, day, month, year):
        timestamped_load = []
        for c, load in enumerate(day_load):
            timestamp = year + '-' + self.num2str(month) + '-' + self.num2str(day) + ' '
            timestamp += self.extract_daytime(c)
            timestamped_load.append((timestamp, load))
        return timestamped_load

    def extract_daytime(self, index):
        hour = int(index / 4)
        subhour = index % 4 * 15
        daytime = self.num2str(hour) + ':' + self.num2str(subhour) + ':00'
        return daytime

    def extract_data_from_files(self):
        weekstart = 2
        for y in self.years:
            os.chdir(y)
            for m in self.months.keys():
                yr = y[-2:]
                os.chdir(m + '-' + yr)
                while self.is_date_in_month(weekstart, m, y):
                    weekend = self.get_weekend(weekstart, m, y)
                    monthnum = self.months[m][0]
                    weekfolder = self.num2str(weekstart) + '.' + self.num2str(monthnum) + '.' + y[-2:] + ' to '
                    yr = y
                    if weekend < weekstart:
                        monthnum = self.months[m][0] + 1
                        if monthnum == 13:
                            monthnum = 1
                            yr = str(int(y) + 1)
                    weekfolder += self.num2str(weekend) + '.' + self.num2str(monthnum) + '.' + yr[-2:] + '_UI_Bills'
                    print(weekfolder)
                    os.chdir(weekfolder)
                    for i in range(0, 7):
                        day = weekstart + i
                        day = self.get_date(day, m, y)
                        monthnum = self.months[m][0]
                        yr = y
                        if day < weekstart:
                            monthnum = self.months[m][0] + 1
                            if monthnum == 13:
                                monthnum = 1
                                yr = str(int(y) + 1)
                        filename = 'UI_BILL_' + str(day) + '-' + str(monthnum) + '-' + yr + '.xls'
                        dayload = self.extract_load(filename)
                        self.timestamped_load.extend(self.generate_timestamped_load(
                            dayload, day, self.months[m][0], yr))
                    weekstart += 7
                    os.chdir('../')
                weekstart = self.get_date(weekstart, m, y)
                os.chdir('../')
            os.chdir('../')
        return self.timestamped_load

    def create_CSV(self):
        csv.register_dialect('myExcel', delimiter=',', quotechar='|', skipinitialspace=True)
        with open('SLDC_Load.csv', 'w', newline='') as file:
            writer = csv.writer(file, dialect='myExcel')
            writer.writerow(['YMDHMS', 'Load'])
            for datapoint in self.timestamped_load:
                writer.writerow(datapoint)

if __name__ == '__main__':
    sldc_dx = SLDC_Data_Extractor()
    sldc_dx.extract_data_from_files()
    sldc_dx.create_CSV()