import random
import json

import numpy as np
import matplotlib.pyplot as plt

import feed_back as fb

def main(min_training_cost=0, min_training_mape=0, min_test_cost=0, min_test_mape=0):
    run_network()
    make_plots(min_training_cost, min_training_mape, min_test_cost, min_test_mape)

def run_network():
    random.seed(23456789)
    np.random.seed(23456789)
    net = fb.Network(hidden_nos=15, cache_len=20, replace_anomalies=False)
    net.SGD(eta=0.25, mu=0.0, mini_batch_size=10, epochs=None, eta_steps=10, monitor_session=True)

def make_plots(min_training_cost, min_training_mape, min_test_cost, min_test_mape):
    f = open("session_monitor_data.json", "r+b")
    training_cost, training_mape, test_cost, test_mape = json.load(f)
    f.close()
    epochs = len(training_cost)
    plot_training_cost(epochs, training_cost, min_training_cost)
    plot_training_mape(epochs, training_mape, min_training_mape)
    plot_test_cost(epochs, test_cost, min_test_cost)
    plot_test_mape(epochs, test_mape, min_test_mape)
    plot_compare_cost(epochs, training_cost, test_cost, min(min_training_cost, min_test_cost))
    plot_compare_mape(epochs, training_mape, test_mape, min(min_training_mape, min_test_mape))

def plot_training_cost(epochs, training_cost, xmin):
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.plot(1+np.arange(xmin, epochs), training_cost[xmin:])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Cost')
    ax.set_title('Cost on Training Data')
    plt.show()

def plot_training_mape(epochs, training_mape, xmin):
    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    ax.plot(1+np.arange(xmin, epochs), training_mape[xmin:])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MAPE')
    ax.set_title('MAPE on Training Data')
    plt.show()

def plot_test_cost(epochs, test_cost, xmin):
    fig = plt.figure(3)
    ax = fig.add_subplot(111)
    ax.plot(1+np.arange(xmin, epochs), test_cost[xmin:])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Cost')
    ax.set_title('Cost on Test Data')
    plt.show()

def plot_test_mape(epochs, test_mape, xmin):
    fig = plt.figure(4)
    ax = fig.add_subplot(111)
    ax.plot(1+np.arange(xmin, epochs), test_mape[xmin:])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MAPE')
    ax.set_title('MAPE on Test Data')
    plt.show()

def plot_compare_cost(epochs, training_cost, test_cost, xmin):
    fig = plt.figure(5)
    ax = fig.add_subplot(111)
    ax.plot(1+np.arange(xmin, epochs), training_cost[xmin:],
            label='Training Date Cost')
    ax.plot(1+np.arange(xmin, epochs), test_cost[xmin:],
             label='Test Data Cost')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Cost')
    ax.set_title('Comparison of Cost on Training Data vs. Test Data')
    plt.legend(loc='upper right')
    plt.show()

def plot_compare_mape(epochs, training_mape, test_mape, xmin):
    fig = plt.figure(6)
    ax = fig.add_subplot(111)
    ax.plot(1+np.arange(xmin, epochs), training_mape[xmin:],
            label='Training Data MAPE')
    ax.plot(1+np.arange(xmin, epochs), test_mape[xmin:],
            label='Test Data MAPE')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MAPE')
    ax.set_title('Comparison of MAPE on Training Data vs. Test Data')
    plt.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    main(min_training_cost=0, min_training_mape=0, min_test_cost=0, min_test_mape=0)