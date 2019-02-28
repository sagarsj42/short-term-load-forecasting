import vector_norm as vn
import probability_distribution_function as pdf

class AnomalyTreaterStats(object):
    def __init__(self):
        vno = vn.VectorNorm()
        vno.find_anomalies()
        vn_an = vno.get_anomalies()
        pdfo = pdf.PDF()
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

if __name__ == "__main__":
    ats = AnomalyTreaterStats()
    ats.print_anomalies()