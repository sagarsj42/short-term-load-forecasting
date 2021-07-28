'''Tutorial to determine the distribution of a randomly generated data.'''

import scipy.stats as sts
from matplotlib import pyplot as plt

class Fit_PDF(object):
    def __init__(self):
        self.distributions = ['norm', 'lognorm', 'expon', 'f', 'weibull_min', 'weibull_max', 't',
                              'rdist', 'logistic', 'gamma']
        self.params = {}
        self.results = []
        self.pvalue = 0
        self.distribution_fit = ""
        self.isfit = False

    def fit(self, data):
        for distri in self.distributions:
            dist = getattr(sts, distri)
            param = dist.fit(data)
            d, pvalue = sts.kstest(data, distri, args=param)
            self.params[distri] = param
            self.results.append((distri, pvalue))

        self.distribution_fit, self.pvalue = max(self.results, key = lambda x:x[1])
        self.isfit = True
        print("Selected distribution: " + self.distribution_fit)
        return self.distribution_fit, self.pvalue

    def generate_random(self, n=1):
        if self.isfit:
            dist = getattr(sts, self.distribution_fit)
            data_gen = dist.rvs(6.3078, size=n)
        else:
            raise ValueError("fit(data) must be called first.")
        return data_gen

    def plot_PDF(self, input):
        self.fit(input)
        data = self.generate_random(n=len(input))
        plt.hist(input, alpha=0.5, label='Given Data')
        plt.hist(data, alpha=0.5, label='Fitted')
        plt.show()

if __name__ == "__main__":
    fitter = Fit_PDF()
    import pandas as pd
    data = pd.read_csv("LD2011_2014_total.csv")
    lower = 365*96 - 1
    upper = 731*96 - 1
    data = data.Load_kW[lower: upper]
    dis_fit, pval = fitter.fit(data)
    print("Fit: " + str(dis_fit) + "\tp-value: " + str(pval))
    print("Params: " + str(fitter.params[dis_fit]))
    fitter.plot_PDF(data)