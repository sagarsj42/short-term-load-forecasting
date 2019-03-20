import random
import json

import numpy as np

import prep_data

class Network(object):
    def __init__(self, hidden_nos=10, cache_len=10, replace_anomalies=False):
        self.training_data, self.test_data = prep_data.DataLoader(replace_anomalies=replace_anomalies).get_data()
        self.num_layers = 3
        ip_len = len(self.training_data[0][0])
        self.sizes = [ip_len, hidden_nos, 1]
        self.weights = [np.random.randn(m, n) for m, n in zip(self.sizes[1:], self.sizes[:-1])]
        self.biases = [np.random.randn(m, 1) for m in self.sizes[1:]]
        self.velocities = [np.zeros(w.shape) for w in self.weights]
        self.best_weights = []
        self.best_biases = []
        self.best_error = 1000
        self.best_epoch = 0
        self.training_cost = []
        self.training_mape = []
        self.test_cost = []
        self.test_mape = []
        self.mape_cache = 100*np.ones(cache_len)

    def SGD(self, mini_batch_size=10, eta=0.25, mu=0.0, epochs=30, eta_steps=10, monitor_session=False):
        if not epochs:
            epoch = 1
            eta_org = eta
            lim = np.power(2, eta_steps)
            while eta > eta_org/lim:
                mini_batches = self.form_mini_batches(mini_batch_size)
                for mini_batch in mini_batches:
                    self.update_mini_batch(mini_batch, eta, mu)
                mape = self.evaluate_mape(self.test_data)
                self.check_acc_n_report(mape, epoch)
                if self.change_eta(mape):
                    eta /= 2
                if monitor_session:
                    self.find_report_other_data()
                    self.test_mape.append(mape)
                epoch += 1
        else:
            for e in range(1, epochs+1, 1):
                mini_batches = self.form_mini_batches(mini_batch_size)
                for mini_batch in mini_batches:
                    self.update_mini_batch(mini_batch, eta, mu)
                mape = self.evaluate_mape(self.test_data)
                self.check_acc_n_report(mape, e)
                if monitor_session:
                    self.find_report_other_data()
                    self.test_mape.append(mape)
        self.store_session_info(monitor_session)

    def find_report_other_data(self):
        training_cost = self.evaluate_cost(self.training_data)
        self.training_cost.append(training_cost)
        training_mape = self.evaluate_mape(self.training_data)
        self.training_mape.append(training_mape)
        test_cost = self.evaluate_cost(self.test_data)
        self.test_cost.append(test_cost)
        print(" Training Data Cost = {0}".format(training_cost))
        print(" Training Data MAPE = {0}% \t Accuracy = {1}%".format(training_mape, 100-training_mape))
        print(" Test Data Cost = {0}".format(test_cost))

    def change_eta(self, mape):
        self.mape_cache = np.roll(self.mape_cache, -1)
        self.mape_cache[-1] = mape
        if self.best_error < np.min(self.mape_cache):
            return True
        return False

    def evaluate_cost(self, data):
        n = len(data)
        cost = 0.0
        for t in data:
            a = feedforward(t[0], self.weights, self.biases)[0][0]
            y = t[1][0][0]
            cost += self.find_quadratic_cost(a, y)
        cost /= n
        return cost

    def find_quadratic_cost(self, a, y):
        return (a - y) ** 2 / 2.0

    def check_acc_n_report(self, error, epoch):
        if self.best_error > error:
            self.best_error = error
            self.best_weights = self.weights
            self.best_biases = self.biases
            self.best_epoch = epoch + 1
        print("\nEpoch {0}: \n Test Data MAPE = {1}% \t Accuracy = {2}%".format(epoch, error, 100 - error))

    def evaluate_mape(self, data):
        n = len(data)
        mape = 0.0
        for t in data:
            actual = t[1][0][0]
            predicted = feedforward(t[0], self.weights, self.biases)[0][0]
            mape += abs(actual - predicted) / actual
            #mape += np.power(actual - predicted, 2)
        mape = mape / n * 100
        #mape = np.power(mape / n, 0.5) * 100
        return mape

    def update_mini_batch(self, mini_batch, eta, mu):
        x = mini_batch[0][0]
        y = mini_batch[0][1]
        for packet in mini_batch[1:]:
            x = np.hstack((x, packet[0]))
            y = np.hstack((y, packet[1]))
        del_w, del_b = self.backpropagate(x, y)
        self.velocities = [mu*v - eta*dw for v, dw in zip(self.velocities, del_w)]
        self.weights = [w + v for w, v in zip(self.weights, self.velocities)]
        self.biases = [b - eta * db for b, db in zip(self.biases, del_b)]

    def backpropagate(self, x, y):
        activation = x
        activations = [x]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        del_w = [np.zeros(w.shape) for w in self.weights]
        del_b = [np.zeros(b.shape) for b in self.biases]
        delta = cost_derivative(activation, y) * sigmoid_prime(zs[-1])
        del_w[-1] = np.dot(delta, activations[-2].transpose())
        del_b[-1] = np.dot(delta, np.ones((delta.shape[1], 1)))
        for lay in range(2, self.num_layers):
            delta = np.dot(self.weights[-lay + 1].transpose(), delta) * sigmoid_prime(zs[-lay])
            del_w[-lay] = np.dot(delta, activations[-lay-1].transpose())
            del_b[-lay] = np.dot(delta, np.ones((delta.shape[1], 1)))
        return del_w, del_b

    def form_mini_batches(self, size):
        random.shuffle(self.training_data)
        mini_batches = [self.training_data[k : k + size] for k in range(0, len(self.training_data), size)]
        return mini_batches

    def store_session_info(self, monitor_session=False):
        np.savez("best_params.npz", weights=self.best_weights, biases=self.best_biases,
                      best_error=self.best_error)
        if monitor_session:
            f = open("session_monitor_data.json", "w")
            json.dump([self.training_cost, self.training_mape, self.test_cost, self.test_mape], f)
            f.close()
        print("\nSession complete. \nBest error = {0}% & best accuracy = {1}% at epoch {2}.".format(
            self.best_error, 100-self.best_error, self.best_epoch))

def feedforward(x, weights, biases):
    for w, b in zip(weights, biases):
        x = sigmoid(np.dot(w, x) + b)
    return x

def cost_derivative(a, y):
    return a - y

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


if __name__ == "__main__":
    net = Network(hidden_nos=16, cache_len=15, replace_anomalies=True)
    net.SGD(mini_batch_size=10, eta=0.3, epochs=5, eta_steps=15, mu=0.0)