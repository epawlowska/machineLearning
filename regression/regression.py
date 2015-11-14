import random
import matplotlib.pyplot as plt
import pandas
import numpy as np
import scipy
import statistics as stat
import math

SQUARED = "squared"


def read_data(file_name):
    cvs_data = pandas.read_csv(file_name)
    data = np.array(cvs_data)
    return data


def hash_trick(x, n):
    new_x = np.zeros((x.shape[0], n + 1), dtype=np.float)
    new_x[:, 0] = 1
    for i in range(x.shape[0]):
        for j in range(1, x.shape[1]):
            hash = x[i, j]
            new_x[i, hash % n + 1] += hash
    return new_x


def standardize(x):
    if x.shape[0] == 0:
        return
    for i in range(1, x.shape[1]):
        variance = stat.variance(x[:, i])
        mean = stat.mean(x[:, i])
        if variance == 0:
            break
        x[:, i] = (x[:, i] - mean)/variance


class Regression:

    def __init__(self, opts):
        self.train_file = opts["train_file"]
        self.test_file = opts["test_file"]
        self.out_file = opts["out_file"]

        self.learning_rate = opts["learning_rate"]
        self.decay_rate = opts["decay_rate"]
        self.batch_size = opts["batch_size"]
        self.n_iter = opts["n_iter"]
        self.shuffle = opts["shuffle"]
        self.holdout_size = opts["holdout_size"]
        self.l2 = opts["l2"]
        self.standardization = opts["standardize"]
        self.loss_method = opts["loss"]
        self.use_adagrad = opts["adagrad"]
        self.use_rmsprop = opts["rmsprop"]
        self.hash_trick_mod = opts["hash"]

        print opts

        train_data = read_data(self.train_file)
        test_data = read_data(self.test_file)

        self.test_input = np.ones((test_data.shape[0], test_data.shape[1] + 1), dtype=np.float)
        self.test_input[:, 1:] = test_data[:, :]
        self.test_initial = np.array(self.test_input)
        self.test_output = np.zeros(test_data.shape[0])

        self.input = np.ones(train_data.shape, dtype=np.float)
        self.input[:, 1:] = train_data[:, :-1]
        self.output = train_data[:, -1:].transpose(1, 0)[0]

        self.validation_input = np.array([])
        self.validation_output = np.array([])
        if self.holdout_size:
            holdout_part = int(self.holdout_size * self.input.shape[0])
            random_rows = random.sample(range(self.input.shape[0]), holdout_part)
            self.validation_input = self.input[random_rows, :]
            self.validation_output = self.output[random_rows]
            self.input = scipy.delete(self.input, random_rows, 0)
            self.output = scipy.delete(self.output, random_rows)

        self.learning_input = np.array(self.input)
        self.learning_output = np.array(self.output)

        if self.hash_trick_mod != 0:
            self.learning_input = hash_trick(self.learning_input, self.hash_trick_mod)
            self.validation_input = hash_trick(self.validation_input, self.hash_trick_mod)
            self.test_input = hash_trick(self.test_input, self.hash_trick_mod)

        if self.standardization:
            standardize(self.learning_input)
            standardize(self.validation_input)
            standardize(self.test_input)

        self.w = np.zeros(self.learning_input.shape[1], dtype=np.float)
        self.adagrad_cache = np.zeros(len(self.w))
        self.rmsprop_cache = np.zeros(len(self.w))

    def h(self, x_row):
        if self.loss_method == SQUARED:
            return x_row.dot(self.w)
        logistic_ret = 1.0/(1 + math.exp(-1 * x_row.dot(self.w)))
        logistic_ret = 0.999999999 if logistic_ret == 1 else logistic_ret
        logistic_ret = 0.000000001 if logistic_ret == 0 else logistic_ret
        return logistic_ret

    def loss_in_row(self, x_row, y_val):
        if self.loss_method == SQUARED:
            return (self.h(x_row) - y_val)**2
        logistic_ret = -(y_val * math.log(self.h(x_row)) + (1 - y_val) * math.log(1 - self.h(x_row)))
        return logistic_ret

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
            ret[i] = self.w[i] - self.learning_rate * self.gradient(x, y, i) / (a * r)
        return ret

    def learn(self):
        learning_input = self.learning_input
        learning_output = self.learning_output

        for epoch in range(self.n_iter):

            train_loss1 = self.loss(learning_input, learning_output)

            if self.batch_size == 0:
                self.batch_size = len(learning_input)

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

    def test(self):
        for i in range(len(self.test_output)):
            self.test_output[i] = self.h(self.test_input[i])

    def print_test_result(self):
        df = pandas.DataFrame(self.test_output)
        df.to_csv(self.out_file)

    def show_train_results(self):
        b = plt.scatter(self.input[:, 1], self.output, color='blue')
        learning_predictions = np.zeros(len(self.input), dtype=np.float)
        for i in range(len(learning_predictions)):
            learning_predictions[i] = self.h(self.learning_input[i])
        r = plt.scatter(self.input[:, 1], learning_predictions, color='red')
        y = plt.scatter(self.test_initial[:, 1], self.test_output, color='yellow')

        plt.legend((b, r, y),
                   ("training sample", "trained model", "test predictions"),
                   scatterpoints=1,
                   loc="lower left",
                   ncol=3,
                   fontsize=8
                   )

        plt.show()