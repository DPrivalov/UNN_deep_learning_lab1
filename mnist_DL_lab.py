import random
import os
import gzip
import cloudpickle
import wget
import argparse

import numpy as np
import tqdm as tqdm
from sklearn.metrics import *

def load_mnist_dataset():
    def vectorized_result(j):
        """Return a 10-dimensional unit vector with a 1.0 in the jth
        position and zeroes elsewhere.  This is used to convert a digit
        (0...9) into a corresponding desired output from the neural
        network."""
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e

    if not os.path.exists(os.path.join(os.curdir, 'data')):
        print ('MNIST downloading from http://deeplearning.net/data/mnist/mnist.pkl.gz')
        os.mkdir(os.path.join(os.curdir, 'data'))
        wget.download('http://deeplearning.net/data/mnist/mnist.pkl.gz', out='data')
    else:
        print ('MNIST data already downloaded')

    with gzip.open(os.path.join(os.curdir, 'data', 'mnist.pkl.gz'), 'rb') as data_file:
        tr_d, va_d, te_d = cloudpickle.load(data_file, encoding='latin1')

    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    _training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    _validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    _test_data = list(zip(test_inputs, te_d[1]))
    return (_training_data, _validation_data, _test_data)


class Network(object):

    def __init__(self, sizes=[784, 392, 10]):
        # Input layer is layer 0, followed by hidden layers layer 1, 2, 3...
        self.sizes = sizes
        self.num_layers = len(sizes)
        print ('Network created with \nLayer sizes =', self.sizes)

        # weights between all layers
        self.weights = [np.array([0])] + [np.random.randn(y, x) for y, x in
                                           zip(sizes[1:], sizes[:-1])] # x, y, z  (z по  x, y по y, x по z)
        # shifts for each neuron
        self.shifts = [np.random.randn(y, 1) for y in sizes]

        # sums of inputs for each neuron
        self.sums = [np.zeros(shift.shape) for shift in self.shifts]
        # activation func applied to sums for each neuron
        self.activations = [np.zeros(shift.shape) for shift in self.shifts]

        # for b in self.activations:
        # print(b.shape)

        print('Network: initialised')

    @staticmethod
    def sigmoid(s):
        # Activation function
        return 1.0 / (1.0 + np.exp(-s))

    def predict(self, x):
        # Set input values
        self.activations[0] = x
        # Forward propagation of values
        for i in range(1, self.num_layers):
            self.sums[i] = (self.weights[i].dot(self.activations[i - 1]) + self.shifts[i])
            self.activations[i] = self.sigmoid(self.sums[i])
        # Finding the most plausible value
        return np.argmax(self.activations[-1])

    def fit(self, data, learning_rate=0.05, mini_batch_size=1, epochs=10):
        print ('Network.fit: started with parameters', \
               '\nLearning rate =', learning_rate, \
               '\nBatch size =', mini_batch_size, \
               '\nEpochs number =', epochs)
        _training_data, _validation_data = data
        for epoch in range(epochs):
            print ('\nEpoch', epoch + 1)
            random.shuffle(_training_data)
            mini_batches = [
                _training_data[k:k + mini_batch_size] for k in
                range(0, len(_training_data), mini_batch_size)]
            for mini_batch_number in tqdm.trange(len(mini_batches), ascii='True',desc='Network Fit iteration'):
                nabla_shifts = [np.zeros(shift.shape) for shift in self.shifts]
                nabla_weights = [np.zeros(weight.shape) for weight in self.weights]
                # if mini_batch_number % 5000 == 0:
                #     print "Accuracy", sum(result for result in [(self.predict(x) == y) for x, y in _validation_data]) / 100.0

                for x, y in mini_batches[mini_batch_number]:
                    # Do FORWARD propagation
                    # Set input values
                    self.activations[0] = x
                    # Forward propagation of values
                    for i in range(1, self.num_layers):
                        self.sums[i] = (self.weights[i].dot(self.activations[i - 1]) + self.shifts[i])
                        self.activations[i] = self.sigmoid(self.sums[i])

                    # Do BACKWARD propagation
                    # Initialize correction
                    delta_nabla_shifts = [np.zeros(shift.shape) for shift in self.shifts]
                    delta_nabla_weights = [np.zeros(weight.shape) for weight in self.weights]

                    # error for output layer ((network_res - right_res) * network_res)
                    error = (self.activations[self.num_layers - 1] - y) * self.sigmoid(self.sums[-1])

                    # Calculate correction for output layer
                    delta_nabla_shifts[-1] = error
                    delta_nabla_weights[-1] = error.dot(self.activations[-2].transpose())

                    # Calculate correction for other layers
                    for l in range(self.num_layers - 2, 0, -1):
                        error = np.multiply(self.weights[l + 1].transpose().dot(error), self.sigmoid(self.sums[l]))
                        # error.reshape()
                        delta_nabla_shifts[l] = error
                        # print(error.shape)
                        delta_nabla_weights[l] = error.dot(self.activations[l - 1].transpose())

                    nabla_shifts = [nb + dnb for nb, dnb in zip(nabla_shifts, delta_nabla_shifts)]
                    nabla_weights = [nw + dnw for nw, dnw in zip(nabla_weights, delta_nabla_weights)]

                # Apply correction for weights and shifts
                self.weights = [
                    w - (learning_rate / mini_batch_size) * dw for w, dw in
                    zip(self.weights, nabla_weights)]
                self.shifts = [
                    b - (learning_rate / mini_batch_size) * db for b, db in
                    zip(self.shifts, nabla_shifts)]

            print ("Accuracy", sum(result for result in [(self.predict(x) == y) for x, y in _validation_data]) / 100)


def get_statistics(network, test_data):
    return classification_report(np.array([y for _, y in test_data]),
                                 np.array([network.predict(x) for x, _ in test_data]),
                                 target_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=0.05,
        type=float)
    parser.add_argument(
        "-bs",
        "--batch_size",
        default=1,
        type=int)
    parser.add_argument(
        "-e",
        "--epochs",
        default=10,
        type=int)
    parser.add_argument(
        "-s",
        "--sizes",
        default='784,392,10',
        type=str)

    return parser


if __name__ == "__main__":
    parser = createParser()
    args = parser.parse_args()

    training_data, validation_data, test_data = load_mnist_dataset()

    network = Network(list(map(int, str(args.sizes).split(','))))

    network.fit((training_data, validation_data),
                learning_rate=args.learning_rate,
                mini_batch_size=args.batch_size,
                epochs=args.epochs)

    print (get_statistics(network, test_data))