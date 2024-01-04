# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 15:47:07 2023

@author: northroj
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

def read_data( file_name, num_train):
    data = np.loadtxt(file_name, delimiter=',')
    x_data = data[1:num_train, 1:81]
    y_data = data[1:num_train, 81:]
    x_val = data[num_train:, 1:81]
    y_val = data[num_train:, 81:]
    print("Data loaded")
    return x_data, y_data, x_val, y_val

def normalize_data():
    for i in range(np.size(y_data, 0)):
        y_data[i, :10] = y_data[i, :10] / np.max(y_data[i, :10])
        y_data[i, 10:20] = y_data[i, 10:20] / np.max(y_data[i, 10:20])
    for i in range(np.size(y_val, 0)):
        y_val[i, :10] = y_val[i, :10] / np.max(y_val[i, :10])
        y_val[i, 10:20] = y_val[i, 10:20] / np.max(y_val[i, 10:20])
    print("Data normalized")

def plot_data( test_case ):
    x_cells = np.arange(1,11,1)
    
    plt.figure(1)
    plt.plot(x_cells, testflux[test_case,:10], 'r')
    plt.plot(x_cells, y_val[test_case,:10], 'b')
    plt.xlabel("cell #")
    plt.ylabel('flux')
    plt.title('fast flux')
    plt.legend(["predicted","calculated"])

    plt.figure(2)
    plt.plot(x_cells,testflux[test_case,10:20], 'r')
    plt.plot(x_cells, y_val[test_case,10:20], 'b')
    plt.xlabel("cell #")
    plt.ylabel('flux')
    plt.title('thermal flux')
    plt.legend(["predicted","calculated"])


[x_data,y_data,x_val,y_val] = read_data('testdata2000.csv', 6500)

#normalize_data()

model1 = MLPRegressor(hidden_layer_sizes=(80,80), learning_rate_init=0.001, tol=1e-6, activation='relu', max_iter=1000, alpha=0.0001, shuffle=True).fit(x_data, y_data)

train_r2 = model1.score(x_data, y_data)
val_r2 = model1.score(x_val, y_val)
testflux = model1.predict(x_val)
kr2 = r2_score(y_val[:,20], testflux[:,20])
ffluxr2 = r2_score(y_val[:,:10], testflux[:,:10])
tfluxr2 = r2_score(y_val[:,10:20], testflux[:,10:20])

plot_data(10)


