# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def sigmoid(x):
    h = 1.0 / (1.0 + np.exp(-x))
    return h


def cost_func(tha, x, y):
    m = y.size
    h = sigmoid(x.dot(tha))
    h = np.c_[h]
    J = -1.0 * (1.0 / m) * (np.log(h).T.dot(y) + np.log(1 - h).T.dot(1 - y))
    return J


def gradient(tha, x, y):
    m = y.size
    h = sigmoid(x.dot(tha))
    h = np.c_[h]
    grad = (1.0 / m) * x.T.dot(h - y)
    return (grad.flatten())


def grad_desc(tha, x, y, lr=0.01, nps=1500):
    # m = y.size
    cost_val = np.zeros(nps)
    for i in range(nps):
        tha = tha - lr * np.c_[gradient(tha, x, y)]
        cost_val[i] = cost_func(tha, x, y)
    cost_v = cost_val[::1000]
    return cost_v, tha


def plotData(data, label_x, label_y, label_pos, label_neg, axes=None):
    # 获得正负样本的下标(即哪些是正样本，哪些是负样本)
    neg = data[:, 2] == 0
    pos = data[:, 2] == 1
    if axes is None:
        axes = plt.gca()
    axes.scatter(data[pos][:, 0], data[pos][:, 1], marker='+',
                 c='k', s=60, linewidth=2, label=label_pos)
    axes.scatter(data[neg][:, 0], data[neg][:, 1],
                 c='y', s=60, label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon=True, fancybox=True)


def ex1():
    data = np.loadtxt('./data1.txt', delimiter=',')
    x, y = data[:, 0:2], data[:, 2]
    # x0 := 1
    xi = np.c_[np.ones((x.shape[0], 1)), x]
    yi = np.c_[y]

    # diffience between learning rates
    lr = 0.0005

    # number of gradient descent
    nps = 5000000

    tha = np.zeros((xi.shape[1], 1))

    # cost value fun: 0.20349770158944375
    res = minimize(cost_func, tha, args=(xi, yi),
                   jac=gradient, options={'maxiter': 400})
    print 'res', res
    # gradient desc
    cost_val, tha = grad_desc(tha, xi, yi, lr, nps=nps)
    print 'tha:', tha
    # save Cost Function Figure
    plt.figure(1)
    cost_x = np.linspace(0, 10, nps / 1000)
    plt.plot(cost_x, cost_val)
    plt.axis([0, 10, 0.2, 0.8])
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Cost Function")
    plt.savefig("CostFunciton")

    # plot data and board
    plt.figure(2)
    plt.scatter(45, 85, s=60, c='r', marker='v', label='(45, 85)')
    plotData(data, 'Exam 1 score', 'Exam 2 score',
             'Admitted', 'Not admitted')
    x1_min, x1_max = xi[:, 1].min(), xi[:, 1].max(),
    x2_min, x2_max = xi[:, 2].min(), xi[:, 2].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
                           np.linspace(x2_min, x2_max))

    h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)),
                      xx1.ravel(), xx2.ravel()].dot(tha))
    h = h.reshape(xx1.shape)
    gd = plt.contour(xx1, xx2, h, [0.5], linewidths=2, colors='b')
    plt.clabel(gd, fontsize=10, inline=1, ticks='gd')
    h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)),
                      xx1.ravel(), xx2.ravel()].dot(res.x))
    h = h.reshape(xx1.shape)
    mim = plt.contour(xx1, xx2, h, [0.5], linewidths=2, colors='r')
    plt.clabel(mim, fontsize=10, inline=1, ticks='minimize')
    plt.legend()
    plt.savefig("LRData")
    plt.show()


def main():
    ex1()


if __name__ == '__main__':
    main()
