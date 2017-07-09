# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Flatten,Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import History, EarlyStopping
from keras import regularizers
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing


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


data = np.loadtxt('./data2.txt', delimiter=',')
np.random.shuffle(data)
x, y = data[:, 0:2], data[:, 2]

x_train, y_train = x[:100, :], y[:100]
x_test, y_test = x[101:, ], y[101:]
model = Sequential()
model.add(Dense(4, kernel_regularizer=regularizers.l2(
    0.), input_shape=(2,), activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
rms = RMSprop(lr=0.01)
model.compile(optimizer=rms, loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
hist = model.fit(x_train, y_train, epochs=1000, batch_size=8,
                 shuffle=True, verbose=1, validation_split=0.2)

score = model.evaluate(x_test, y_test, verbose=1)
print(score)
y_pre = model.predict(x_test)
print(np.c_[y_pre, y_test])
figures(hist)
