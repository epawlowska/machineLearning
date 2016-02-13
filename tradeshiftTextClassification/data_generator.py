import pandas
import numpy as np
import random
import statistics as stat
import argparse

def add_arguments(parser):
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--logistic", action="store_true", default=False)

def generate_data(file_name, n=30, a=[1, 2], b=[-5, 0], c=[10, 1], logistic=False, test=False):
    m = min(len(a), len(b), len(c))
    data = np.ones((n, m + 1), dtype=np.float)
    for i in range(n):
        data[i, 0] = random.random()
    for i in range(m):
        data[:, i] = a[i] * data[:, i] + b[i]
    for i in range(n):
        data[i, -1] = sum(c[j] * data[i, j] for j in range(m)) + random.gauss(0, 0.3)
    if logistic:
        mean = stat.mean(data[:, -1])
        for i in range(n):
            data[i, -1] = 1.0 if data[i, -1] > mean else 0.0

    df = pandas.DataFrame(data)
    if test:
        df = pandas.DataFrame(data[:, :-1])

    df.to_csv(file_name, index=False)


parser = argparse.ArgumentParser()
add_arguments(parser)
opts = parser.parse_args()

n = 50
generate_data(opts.train_file, n, logistic=opts.logistic)
generate_data(opts.test_file, 3 * n, test=True, logistic=opts.logistic)