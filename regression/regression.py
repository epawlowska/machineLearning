import random
import matplotlib.pyplot as plt
import pandas
import numpy as np
import scipy
import statistics as stat
import math

import data_generator as dg

SQUARED = "squared"

def get_data_from_csv(file_name):
    cvs_data = pandas.read_csv(file_name)
    data = np.array(cvs_data)
    return data

samples_count = 50
dg.generate_csv_data("train.csv", samples_count, [1, 2, 0], [-5, 0, 0], [10, 1, 3])
dg.generate_test_data("test.csv", 2 * samples_count, [1, 2, 0], [-5, 0, 0], [10, 1, 3])

#14, 0.009, 0,99, without standardization
#30, 099, 0,9999, standarization
#10, 1.0, 1.0, n_iter = 50000, hash_trick, adagrad
#50, 1.0, 0.9999, n_iter = 10000, hash_trick, rmsprob COOL!

class Regression:

    train_file = "train.csv"
    test_file = "test.csv"
    out_file = "out.csv"

    learning_rate = 1.0
    decay_rate = 0.9
    batch_size = 10
    n_iter = 5
    shuffle = True
    holdout_size_coef = 0.3
    l2 = 0.0
    standardization = False #True
    loss_method = SQUARED
    use_adagrad = False
    use_rmsprop = True
    use_hash_trick = True
    ht_mod = 1
    adagrad_cache = None
    rmsprop_cache = None

    initial_input = None
    initial_output = None

    learning_input = None
    learning_output = None

    validation_input = None
    validation_output = None

    test_input = None
    test_initial = None
    test_output = None

    w = None

    def __init__(self):
        self.prepare_data()

    def prepare_data(self): #TODO name

        train_data = get_data_from_csv(self.train_file)
        #dg.generate_test_data(self.test_file, train_data)

        test_data = get_data_from_csv(self.test_file)
        self.test_input = np.ones((test_data.shape[0], test_data.shape[1] + 1), dtype=np.float)
        self.test_input[:, 1:] = test_data[:, :]
        self.test_initial = np.array(self.test_input)
        self.test_output = np.zeros(test_data.shape[0])

        self.initial_input = np.ones(train_data.shape, dtype=np.float)
        self.initial_input[:, 1:] = train_data[:, :-1]
        self.initial_output = train_data[:, -1:].transpose(1, 0)[0]

        self.extract_validation_data()

        self.learning_input = np.array(self.initial_input)
        self.learning_output = np.array(self.initial_output) #not necessary

        if self.use_hash_trick:
            self.learning_input = self.hash_trick(self.learning_input, self.ht_mod)
            self.validation_input = self.hash_trick(self.validation_input, self.ht_mod)
            self.test_input = self.hash_trick(self.test_input, self.ht_mod)

        if self.standardization:
            self.standardize(self.learning_input)
            self.standardize(self.validation_input)
            self.standardize(self.test_input)

        self.w = np.zeros(self.learning_input.shape[1], dtype=np.float)
        self.adagrad_cache = np.zeros(len(self.w))
        self.rmsprop_cache = np.zeros(len(self.w))

    def extract_validation_data(self):
        holdout_size = int(self.holdout_size_coef * self.initial_input.shape[0])
        random_rows = random.sample(range(self.initial_input.shape[0]), holdout_size)
        self.validation_input = self.initial_input[random_rows, :]
        self.validation_output = self.initial_output[random_rows]
        self.initial_input = scipy.delete(self.initial_input, random_rows, 0)
        self.initial_output = scipy.delete(self.initial_output, random_rows)

    def hash_trick(self, x, n):
        new_x = np.zeros((x.shape[0], n + 1), dtype=np.float)
        new_x[:, 0] = 1
        for i in range(x.shape[0]):
            for j in range(1, x.shape[1]):
                hash = x[i, j]
                new_x[i, hash % n + 1] += hash
        return new_x

    def standardize(self, x):
        for i in range(1, x.shape[1]):
            variance = stat.variance(x[:, i])
            mean = stat.mean(x[:, i])
            if variance == 0:
                break
            x[:, i] = (x[:, i] - mean)/variance

    def h(self, x_row):
        if self.loss_method == SQUARED:
            return x_row.dot(self.w)
        return 1.0/(1 + math.exp(-1 * x_row.dot(self.w)))

    def loss_in_row(self, x_row, y_val):
        if self.loss_method == SQUARED:
            return (self.h(x_row) - y_val)**2
        return -(y_val * math.log(self.h(x_row)) + (1 - y_val) * math.log(1 - self.h(x_row)))

    def loss(self, x, y):
        if x.shape[0] == 0:
            return 0
        ret = 0
        for i in range(x.shape[0]):
            ret += self.loss_in_row(x[i], y[i])
        l2_elem = self.l2 * sum(self.w[j]**2 for j in range(1, len(self.w)))
        return ret/(2 * x.shape[0]) + l2_elem

    def gradient(self, x, y, point):
        ret = 0.0
        for i in range(x.shape[0]):
            ret += (self.h(x[i]) - y[i]) * x[i, point]
        l2_elem = self.l2 * 2 * self.w[point]
        return ret/x.shape[0] + l2_elem

    def adagrad(self, row, grad):
        self.adagrad_cache[row] += grad**2
        return math.sqrt(self.adagrad_cache[row])

    def rmsprop(self, row, grad):
        self.rmsprop_cache[row] *= self.decay_rate
        self.rmsprop_cache[row] += (1.0 - self.decay_rate) * grad**2
        return math.sqrt(self.rmsprop_cache[row])

    def update_w(self, x, y):
        ret = np.zeros(len(self.w))
        for i in range(len(self.w)):
            grad = self.gradient(x, y, i)
            a = self.adagrad(i, grad) if self.use_adagrad else 1.0
            r = self.rmsprop(i, grad) if self.use_rmsprop else 1.0
            ret[i] = self.w[i] - \
                     self.learning_rate * self.gradient(x, y, i) / (a * r)
        return ret

    def learn(self):
        learning_input = self.learning_input
        learning_output = self.learning_output

        for epoch in range(self.n_iter):

            train_loss1 = self.loss(learning_input, learning_output)

            i_list = list(range(0, len(learning_input), self.batch_size))
            if self.shuffle:
                random.shuffle(i_list)

            for i in i_list:
                end = min(i + self.batch_size, len(learning_input))
                self.w = self.update_w(learning_input[i:end, :], learning_output[i:end])
                self.learning_rate *= self.decay_rate

            train_loss2 = self.loss(learning_input, learning_output)
            valid_loss = self.loss(self.validation_input, self.validation_output)

            print "epoch: ", epoch
            print "train loss: ", train_loss1
            print "valid loss: ", valid_loss
            if train_loss2 == train_loss1:
                print "break after convergence"
                break

    def show_train_results(self):
        #print self.train_data
        plt.scatter(self.initial_input[:, 1], self.initial_output, color='blue')
        output = np.zeros(len(self.initial_input), dtype=np.float)
        for i in range(len(output)):
            output[i] = self.h(self.learning_input[i])
        plt.scatter(self.initial_input[:, 1], output, color='red')
        for i in range(len(self.test_output)):
            self.test_output[i] = self.h(self.test_input[i])
        plt.scatter(self.test_initial[:, 1], self.test_output, color='yellow')
        plt.show()


regression = Regression()
regression.learn()
regression.show_train_results()
