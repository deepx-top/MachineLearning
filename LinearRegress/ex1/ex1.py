# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


# linear cost funciton
def cost_func(x, y, tha=[[0], [0]]):
    m = y.size
    J = (1.0 / (2 * m)) * np.sum(np.square(x.dot(tha) - y))
    return J


def grad_desc(x, y, tha=[[0], [0]], lr=0.01, nps=1500):
    m = y.size
    cost_val = np.zeros(nps)
    for i in range(nps):
        h = x.dot(tha)
        tha = tha - lr * (1.0 / m) * (x.T.dot(h - y))
        cost_val[i] = cost_func(x, y, tha)
    return cost_val, tha


def ex1():
    data = np.loadtxt('./lin_data1.txt', delimiter=',')
    x, y = data[:, 0], data[:, 1]
    xi = np.c_[np.ones(x.shape[0]), x]
    yi = np.c_[y]

    # diffience between learning rates
    lr = [0.005, 0.001, 0.0005]

    # number of gradient descent
    nps = 1500
    # plot diffent learning rates
    cost_x = np.linspace(0, 100, 1500)
    tha_x = np.linspace(0, 100, 100)
    p1 = plt.subplot(2, 1, 1)
    p1.axis([0, 100, 0, 10])
    p1.set_title("Cost Function")
    p2 = plt.subplot(2, 1, 2)
    p2.axis([0, 40, 0, 40])
    p2.set_title("Learner Regression")
    p2.plot(x, y, 'ro')
    for i in range(len(lr)):
        cost_val, tha = grad_desc(xi, yi, lr=lr[i], nps=nps)
        p1.plot(cost_x, cost_val, label=str(lr[i]))
        p2.plot(tha_x, tha[0] + tha[1] * tha_x, label=str(lr[i]))
    p1.legend()
    p2.legend()
    plt.show()


def main():
    ex1()


if __name__ == '__main__':
    main()
