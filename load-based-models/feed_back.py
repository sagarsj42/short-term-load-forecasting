import random
import json

import numpy as np

import prep_data

class Network(object):
    def __init__(self, hidden_nos=1):
        self.training_data, self.test_data = prep_data.DataLoader().get_data()
        self.num_layers = 3
        ip_len = len(self.training_data[0][0])
        self.sizes = [ip_len, hidden_nos, 1]
        self.weights = [np.random.randn(m, n) for m, n in zip(self.sizes[1:], self.sizes[:-1])]
        self.biases = [np.random.randn(m, 1) for m in self.sizes[1:]]
        self.best_weights = []
        self.best_biases = []
        self.best_acc = 0
        self.best_epoch = 0
        self.training_cost = []
        self.training_mape = []
        self.test_cost = []
        self.test_mape = []

    def feedforward(self, x):
        for w, b in zip(self.weights, self.biases):
            x = self.sigmoid(np.dot(w, x) + b)
        return x

    def SGD(self, epochs=20, mini_batch_size=10, eta=0.5, monitor_session=False):
        for e in range(epochs):
            mini_batches = self.form_mini_batches(mini_batch_size)
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            mape = self.evaluate_mape(self.test_data)
            self.check_acc_n_report(mape, e)
            if monitor_session:
                self.training_cost.append(self.evaluate_cost(self.training_data))
                self.training_mape.append(self.evaluate_mape(self.training_data))
                self.test_cost.append(self.evaluate_cost(self.test_data))
                self.test_mape.append(mape)
        self.store_session_info()

    def evaluate_cost(self, data):
        n = len(data)
        cost = 0.0
        for t in data:
            a = self.feedforward(t[0])[0][0]
            y = t[1][0][0]
            cost += self.find_quadratic_cost(a, y)
        cost /= n
        return cost

    def find_quadratic_cost(self, a, y):
        return (a - y) ** 2 / 2.0

    def check_acc_n_report(self, mape, epoch):
        acc = 100 - mape
        if self.best_acc < acc:
            self.best_acc = acc
            for w, b in zip(self.weights, self.biases):
                self.best_weights.append(w.tolist())
                self.best_biases.append(b.tolist())
            self.best_epoch = epoch + 1
        print("Epoch {0}: \n MAPE = {1}% \t Accuracy = {2}%".format(epoch + 1, mape, acc))

    def evaluate_mape(self, data):
        n = len(data)
        mape = 0.0
        for t in data:
            actual = t[1][0][0]
            predicted = self.feedforward(t[0])[0][0]
            mape += abs(actual - predicted) / actual
        mape = mape / n * 100
        return mape

    def update_mini_batch(self, mini_batch, eta):
        x = mini_batch[0][0]
        y = mini_batch[0][1]
        for packet in mini_batch[1:]:
            x = np.hstack((x, packet[0]))
            y = np.hstack((y, packet[1]))
        del_w, del_b = self.backpropagate(x, y)
        self.weights = [w - eta * dw for w, dw in zip(self.weights, del_w)]
        self.biases = [b - eta * db for b, db in zip(self.biases, del_b)]

    def backpropagate(self, x, y):
        activation = x
        activations = [x]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        del_w = [np.zeros(w.shape) for w in self.weights]
        del_b = [np.zeros(b.shape) for b in self.biases]
        delta = self.cost_derivative(activation, y) * self.sigmoid_prime(zs[-1])
        del_w[-1] = np.dot(delta, activations[-2].transpose())
        del_b[-1] = np.dot(delta, np.ones((delta.shape[1], 1)))
        for lay in range(2, self.num_layers):
            delta = np.dot(self.weights[-lay + 1].transpose(), delta) * self.sigmoid_prime(zs[-lay])
            del_w[-lay] = np.dot(delta, activations[-lay-1].transpose())
            del_b[-lay] = np.dot(delta, np.ones((delta.shape[1], 1)))
        return del_w, del_b

    def form_mini_batches(self, size):
        random.shuffle(self.training_data)
        mini_batches = [self.training_data[k : k + size] for k in range(0, len(self.training_data), size)]
        return mini_batches

    def store_session_info(self):
        np.savez("best_params.npz", best_weights=self.best_weights, best_biases=self.best_biases,
                      best_accuracy=self.best_acc)
        f = open("session_monitor_data.json", "w")
        json.dump([self.training_cost, self.training_mape, self.test_cost, self.test_mape], f)
        f.close()
        print("\nSession complete. \nBest accuracy = {0}% at epoch {1}.".format(self.best_acc, self.best_epoch))

    def cost_derivative(self, a, y):
        return a - y

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

if __name__ == "__main__":
    net = Network(10)
    net.SGD(epochs=30, mini_batch_size=15, eta=0.5)