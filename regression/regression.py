import random
import matplotlib.pyplot as plt
import numpy as np
from sympy import *

n = 50
a = -3
b = -7
coef = np.matrix([[-7, -3], [2, 9], [4, 5]])
f = lambda x, row: coef[0, row] + coef[1, row] * x
f_with_noise = lambda x, row: f(x, row) + random.gauss(0, 0.1) # TODO gauss, mu, sigma
f0 = lambda x: coef[0, 0] + coef[1, 0] * x
f0_with_noise = lambda x: f0(x) + random.gauss(0, 0.1)
x_generated = np.array([random.random() for _ in xrange(n)])
#data = np.matrix([[x_generated], [])
y_generated =  np.array(map(f0_with_noise, x_generated))

class Regression:

    learning_rate = 0.5 #0.5
    learning_rate_decay = 0.99 #0.99
    x_input = x_generated

    y_input = y_generated
    batch_size = 6
    n_iter = 10000
    shuffle = True

    def linear(self):
        w = np.array([0.0, 0.0])
        hypothesis = lambda w, x: w[0] + w[1] * x
        #cost_function = lambda w: sum((hypothesis(w, self.x_input) - self.y_input)**2)/(2 * len(self.x_input))
        derivate_0 = lambda w, x, y: sum(hypothesis(w, x) - y)/len(x)
        derivate_1 = lambda w, x, y: sum((hypothesis(w, x) - y) * x)/len(x)

        for _ in range(self.n_iter):
            i_list = list(range(0, len(self.x_input), self.batch_size))
            if self.shuffle:
                random.shuffle(i_list)
            for i in i_list:
                end = min(i + self.batch_size, len(self.x_input) - 1)
                delta_w0 = self.learning_rate * derivate_0(w, self.x_input[i:end], self.y_input[i:end])
                delta_w1 = self.learning_rate * derivate_1(w, self.x_input[i:end], self.y_input[i:end])
                w[0] -= delta_w0
                w[1] -= delta_w1
                self.learning_rate *= self.learning_rate_decay
        print(w)
        self.predictions = w
        plt.scatter(self.x_input, self.y_input, color='red')
        plt.scatter(self.x_input, hypothesis(w, self.x_input), color="green")
        #plt.show()

regression = Regression()
regression.linear()
