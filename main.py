import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
import matplotlib.pyplot as plt


def process_data(data, num_days):
    """
    Takes a time series data and a lookback window
    num_days, and creates input/output pairs for a machine learning model.
    Specifically, it creates X as a numpy array of shape (num_samples, num_days),
    where num_samples is the number of input/output pairs, and Y as a numpy array
    of shape (num_samples,). The X array contains num_days time steps of input data for
    each sample, and the Y array contains the corresponding output values that we
    want to predict.
    param: data - data that is being processed
    param: num_days - lookback window
    return: (np.array(x), np.array(y)) pair
    """
    x, y = [], []
    for i in range(len(data) - num_days - 1):
        x.append(data[i:(i+num_days), 0])
        y.append(data[(i+num_days), 0])
    return np.array(x), np.array(y)

def stock_prediction(stock_symbol):
    """

    """
    data = pd.read_csv('archive/all_stocks_5yr.csv')
    column = data[data['Name'] == stock_symbol].Close
    path = os.getcwd()
    os.mkdir(path+"/archive/stocks/"+stock_symbol)

    scale = MinMaxScaler()  # used to scale the input data to a specified range
    column = column.values.reshape(column.shape[0], 1)  # reshapes the data to have a single column and a number of rows equal to the original shape of the array
    column = scale.fit_transform(column)  # fits the scaler to the data, then transforms the data within a certain range



