import random
import matplotlib.pyplot as plt
import pandas
import numpy as np


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


def cost_function(w, x, y):
    sum = 0
    for i in range(x.shape[0]):
        sum += (x[i].dot(w) - y[i])**2
    return sum/(2 * x.shape[0])


def partial_derivate(w, x, y, point):
    sum = 0.0
    for i in range(x.shape[0]):
        sum += (w.dot(x[i]) - y[i]) * x[i, point]
    return sum/x.shape[0]


def changed_w(w, x, y, learning_rate):
    ret = np.zeros(w.shape)
    for i in range(len(w)):
        ret[i] = w[i] - learning_rate * partial_derivate(w, x, y, i)
    return ret


file_name = "data.csv"
samples_count = 10
generate_csv_data(file_name, samples_count, [1, 2, 0], [-5, 0, 0], [10, 1, 3])
data = get_data_from_csv(file_name)

#20, 009, 0,999

class Regression:

    learning_rate = 0.009
    learning_rate_decay = 0.999

    train_input = np.ones(data.shape, dtype=np.float)
    train_input[:, 1:] = data[:, :-1]
    train_output = data[:, -1:].transpose(1, 0)[0]

    batch_size = 5
    n_iter = 1000
    shuffle = True

    w_initial = np.zeros(train_input.shape[1], dtype=np.float)
    w_predicted = w_initial

    def show_results(self):
        print data
        print(self.w_predicted)
        plt.scatter(self.train_input[:,1], self.train_output, color='blue')
        plt.scatter(self.train_input[:,1], self.train_input.dot(self.w_predicted), color='red')
        plt.show()

    def learn(self):
        w = self.w_initial
        for _ in range(self.n_iter):

            cf1 = cost_function(w, self.train_input, self.train_output)
            print "train loss ", cf1

            i_list = list(range(0, len(self.train_input), self.batch_size))
            if self.shuffle:
                random.shuffle(i_list)

            for i in i_list:
                end = min(i + self.batch_size, len(self.train_input) - 1)
                w = changed_w(w, self.train_input[i:end, :], self.train_output[i:end], self.learning_rate)
                self.learning_rate *= self.learning_rate_decay

            cf2 = cost_function(w, self.train_input, self.train_output)
            if cf2 == cf1:
                break

        self.w_predicted = w

regression = Regression()
regression.learn()
regression.show_results()
