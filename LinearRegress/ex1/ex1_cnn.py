# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import History


def figures(history, figure_name="plots"):
    """ method to visualize accuracies and loss vs epoch for training as well as testind data
        Argumets: history     = an instance returned by model.fit method
                  figure_name = a string representing file name to plots. By default it is set to "plots"
       Usage: hist = model.fit(X,y)
       figures(hist) """
    if isinstance(history, History):
        hist = history.history
        epoch = history.epoch
        acc = hist['acc']
        loss = hist['loss']
        val_loss = hist['val_loss']
        val_acc = hist['val_acc']
        plt.figure(1)

        plt.subplot(2, 2, 1)
        plt.plot(epoch, acc)
        plt.title("Training accuracy vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

        plt.subplot(2, 2, 2)
        plt.plot(epoch, loss)
        plt.title("Training loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.subplot(2, 2, 3)
        plt.plot(epoch, val_acc)
        plt.title("Validation Acc vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")

        plt.subplot(2, 2, 4)
        plt.plot(epoch, val_loss)
        plt.title("Validation loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.tight_layout()
        plt.savefig(figure_name)
    else:
        print("Input Argument is not an instance of class History")


data = np.loadtxt('./lin_data1.txt', delimiter=',')
x, y = data[:, 0], data[:, 1]
x = (x - x.min()) / (x.max() - x.min())
y = (y - y.min()) / (y.max() - y.min())
# x = normalize(x)
# y = normalize(y)
plt.scatter(x, y)
np.random.shuffle(x)
x_train, y_train = x[:80, ], x[:80, ]
x_test, y_test = x[81:, ], y[81:, ]

model = Sequential()
model.add(Dense(10, input_dim=1, init='uniform', activation='relu'))
model.add(Dense(1, activation='tanh'))

model.compile(loss='mse',
              optimizer='sgd', metrics=["accuracy"])
hist = model.fit(x_train, y_train, batch_size=3, nb_epoch=1500,
                 shuffle=True, verbose=0, validation_split=0.2)
score = model.evaluate(x_test, y_test, batch_size=8, verbose=1)
print(score)
y_pre = model.predict(x_test, batch_size=1)
plt.plot(x_test, y_pre, 'k--', lw=2)
plt.plot(x_test, y_test, 'rx')
plt.show()
figures(hist)
