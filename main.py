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

    x, y = process_data(data, 7)
    x_train, x_test = x[:int(x.shape[0]*0.80)], x[int(x.shape[0]*0.80):]
    y_train, y_test = y[:int(y.shape[0]*0.80)], y[int(y.shape[0]*0.80):]
    # split the input/output data into training and testing sets such that the training set
    # contains the first 80% of the data, and the testing set contains the remaining 20%.

    model = Sequential()
    model.add(LSTM(32, input_shape=(7, 1)))  # adds a LSTM layer to the model with 32 units and an input shape of (7, 1).
    model.add(Dropout(0.5))  # randomly sets 50% of input units to 0 to reduce overfitting
    model.add(Dense(1))  # adds a fully connected dense layer with one unit to the model
    model.compile(optimizer='adam', loss='mse')  # compiles model with adam optimizer and mean squared error loss function

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    # reshapes the X_train array to have a shape of (num_samples, num_timesteps, num_features).

    histogram = model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test), shuffle=False)

    plt.plot(histogram.history['loss'])
    plt.plot(histogram.history['val_loss'])
    plt.legend(['Loss', 'Validation Loss'], loc='upper right')
    plt.savefig('archive/stocks/'+stock_symbol+'/'+stock_symbol+'2.png')
    plt.clf()
    plt.close()

    # uses the trained RNN LSTM model to make predictions on the test data, and then evaluates the accuracy
    # of the model's predictions by finding the average percent error between the predicted values and the real values.
    index = 249
    predicted_val = model.predict(x_test[index].reshape(1, 7, 1))  # uses the trained model to make a prediction on a single sample from the test data
    predicted_price = scale.inverse_transform(predicted_val).copy()
    predicted_val = model.predict(x_test)  # uses the trained model to make predictions on all the test data
    real_values = scale.inverse_transform(y_test.reshape(-1, 1))  # scales the actual values back to the original scale
    pred_values = scale.inverse_transform(predicted_val)  # scales the predicted values back to the original scale
    avg_percent_error = 0
    for i in range(len(real_values)):  # loops over all the samples
        # in the test data and computes the average percentage error
        # between the predicted values the actual values
        avg_percent_error += abs((real_values[i] - pred_values[i]) / real_values[i]) * 100
    avg_percent_error = round(float(avg_percent_error / len(real_values)), 2)
    accuracy = 100 - avg_percent_error

