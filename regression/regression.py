import random
import matplotlib.pyplot as plt
import pandas
import numpy as np
import scipy
import statistics as stat


def get_data_from_csv(file_name):
    cvs_data = pandas.read_csv(file_name)
    data = np.array(cvs_data)
    return data


def generate_csv_data(input_file_name, n=30, a=[1, 2, 0], b=[-5, 0, 0], c=[10, 1, 3]):
    m = 5
    data = np.ones((n, m), dtype=np.float)
    for i in range(3):
        f_with_noise = lambda row, col: row * a[i] + b[i]
        data[:, i] = np.fromfunction(f_with_noise, (n, 1)).transpose(1,0)
    for i in range(data.shape[0]):
        data[i, m - 1] = sum(c[j] * data[i, j] for j in range(3)) + 10 * random.gauss(0, 0.7)

    df = pandas.DataFrame(data)
    df.to_csv(input_file_name, index=False)


file_name = "data.csv"
samples_count = 30
generate_csv_data(file_name, samples_count, [1, 2, 0], [-5, 0, 0], [10, 1, 3])
data = get_data_from_csv(file_name)

#20, 009, 0,999, without standardization
#30, 099, 0,9999, standarization

class Regression:

    learning_rate = 0.099
    learning_rate_decay = 0.9999
    batch_size = 5
    n_iter = 10000
    shuffle = True
    holdout_size_coef = 0.0
    l2 = 0.0
    standardization = True

    initial_input = np.ones(data.shape, dtype=np.float)
    initial_input[:, 1:] = data[:, :-1]
    initial_output = data[:, -1:].transpose(1, 0)[0]

    gradient = np.zeros(initial_input.shape[1], dtype=np.float)
    predictor = gradient

    holdout_size = int(holdout_size_coef * initial_input.shape[0])
    random_rows = random.sample(range(initial_input.shape[0]), holdout_size)
    validation_input = initial_input[random_rows, :]
    validation_output = initial_output[random_rows]
    initial_input = scipy.delete(initial_input, random_rows, 0)
    initial_output = scipy.delete(initial_output, random_rows)

    def standardize(self, x):
        for i in range(1, x.shape[1]):
            variance = stat.variance(x[:, i])
            mean = stat.mean(x[:, i])
            if variance == 0:
                break
            x[:, i] = (x[:, i] - mean)/variance

    def loss(self, x, y):
        if x.shape[0] == 0:
            return 0
        ret = 0
        for i in range(x.shape[0]):
            ret += (x[i].dot(self.gradient) - y[i])**2
        l2_elem = self.l2 * sum(self.gradient[j]**2 for j in range(1, len(self.gradient)))
        return ret/(2 * x.shape[0]) + l2_elem #TODO before division?

    def partial_derivate(self, x, y, point):
        ret = 0.0
        for i in range(x.shape[0]):
            ret += (self.gradient.dot(x[i]) - y[i]) * x[i, point]
        l2_elem = self.l2 * 2 * self.gradient[point]
        return ret/x.shape[0] + l2_elem

    def update_gradient(self, x, y):
        ret = np.zeros(len(self.gradient))
        for i in range(len(self.gradient)):
            ret[i] = self.gradient[i] - \
                     self.learning_rate * self.partial_derivate(x, y, i)
        return ret

    def learn(self):
        learning_input = self.initial_input
        learning_output = self.initial_output

        if self.standardization:
            self.standardize(learning_input)

        for epoch in range(self.n_iter):

            train_loss1 = self.loss(learning_input, learning_output)

            i_list = list(range(0, len(learning_input), self.batch_size))
            print i_list
            if self.shuffle:
                random.shuffle(i_list)

            for i in i_list:
                end = min(i + self.batch_size, len(learning_input))
                self.gradient = self.update_gradient(learning_input[i:end, :], learning_output[i:end])
                self.learning_rate *= self.learning_rate_decay

            train_loss2 = self.loss(learning_input, learning_output)
            valid_loss = self.loss(self.validation_input, self.validation_output)

            print "epoch: ", epoch
            print "train loss: ", train_loss1
            print "valid loss: ", valid_loss
            if train_loss2 == train_loss1:
                print "break after convergence"
                break

        print learning_input
        self.predictor = self.gradient

    def show_results(self):
        print data
        print(self.predictor)
        print self.initial_input
        print self.initial_output
        print self.validation_input
        print self.validation_output
        plt.scatter(self.initial_input[:, 1], self.initial_output, color='blue')
        plt.scatter(self.initial_input[:, 1], self.initial_input.dot(self.predictor), color='red')
        plt.show()


regression = Regression()
regression.learn()
regression.show_results()
