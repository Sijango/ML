import math

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

data = pd.read_csv('data.csv', sep=',')
print(data)

data['Price'] = data['Price'].astype('float')
data['Company_index'] = data['Company_index'].astype('float')
data['Supplier_index'] = data['Supplier_index'].astype('float')

# plt.figure(figsize=(21, 7))
# plt.plot(data['Year'].values, data['Price'].values, label='Price', color='black')
# plt.xticks(np.arange(10, data.shape[0], 20))
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.legend()
# plt.show()
#
# plt.figure(figsize=(21, 7))
# plt.plot(data['Year'].values, data['Company_index'].values, label='Company_index', color='red')
# plt.plot(data['Year'].values, data['Supplier_index'].values, label='Supplier_index', color='blue')
# plt.xticks(np.arange(10, data.shape[0], 20))
# plt.xlabel('Date')
# plt.ylabel('Index')
# plt.legend()
# plt.show()

num_train = 240
num_test = 35

train = data.iloc[:num_train, 1:2].values
test = data.iloc[num_train:275, 1:2].values
print(train)
sc = MinMaxScaler(feature_range=(0, 1))
train_sc = sc.fit_transform(train)
test_sc = sc.fit_transform(test)

X_train = []
Y_train = []

X_test_11 = []

window = 4

for i in range(window, num_train):
    X_train_ = np.reshape(train_sc[i-window:i, 0], (window, 1))
    X_train.append(X_train_)
    Y_train.append(train_sc[i, 0])

X_train = np.stack(X_train)
Y_train = np.stack(Y_train)

for i in range(window, num_test):
    X_test_ = np.reshape(test_sc[i - window:i, 0], (window, 1))
    X_test_11.append(X_test_)

X_test_11 = np.stack(X_test_11)

input_shape = X_train.shape[1], 1
print('input_shape={}'.format(input_shape))

model_LSTM = Sequential()

model_LSTM.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
model_LSTM.add(Dropout(0.2))

model_LSTM.add(LSTM(units=32, return_sequences=True))
model_LSTM.add(Dropout(0.2))

model_LSTM.add(LSTM(units=32, return_sequences=True))
model_LSTM.add(Dropout(0.2))

model_LSTM.add(LSTM(units=64))
model_LSTM.add(Dropout(0.2))

model_LSTM.add(Dense(units=1))

model_LSTM.summary()

model_LSTM.compile(optimizer='adam', loss='mean_squared_error')
model_LSTM.fit(X_train, Y_train, epochs=35, batch_size=32)

df_volume = np.vstack((train, test))

inputs = df_volume[df_volume.shape[0] - test.shape[0] - window:]
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

num_2 = df_volume.shape[0] - num_train + window
print(inputs.shape)
print(inputs)
print(num_2, ' num2 !!!!!!!!')
X_test = []

for i in range(window, num_2):
    X_test_ = np.reshape(inputs[i - window:i, 0], (window, 1))
    X_test.append(X_test_)

X_test = np.stack(X_test)
print('X_test.shape={}'.format(X_test.shape))

predict = model_LSTM.predict(X_test)
print('test.shape={}'.format(test.shape), ' ', 'predict.shape={}'.format(predict.shape))
predict = sc.inverse_transform(predict)

predict2 = model_LSTM.predict(X_test_11)
print('test.shape={}'.format(test.shape), ' ', 'predict.shape={}'.format(predict2.shape))
predict2 = sc.inverse_transform(predict2)

diff = predict - test

print("MSE:", np.mean(diff**2))
print("MAE:", np.mean(abs(diff)))
print("RMSE:", np.sqrt(np.mean(diff**2)))


plt.figure(figsize=(20, 7))
plt.plot(data['Year'].values, data['Price'].values, label='Price', color='red')
plt.plot(data['Year'][-predict.shape[0]:].values, predict, label='Predict price', color='Blue')
plt.xticks(np.arange(10, data.shape[0], 20))
plt.xlabel('Date')
plt.ylabel('Price/Predict')
plt.legend()
plt.show()

plt.figure(figsize=(20, 7))
plt.plot(data['Year'].values, data['Price'].values, label='Price', color='red')
plt.plot(data['Year'][-predict2.shape[0]:].values, predict2, label='Predict price', color='Blue')
plt.xticks(np.arange(10, data.shape[0], 20))
plt.xlabel('Date')
plt.ylabel('Price/Predict')
plt.legend()
plt.show()

pred_ = predict[-1].copy()
prediction_full = []
window = 4
df_copy = data.iloc[:, 1:2][1:].values

for j in range(5):
    df_ = np.vstack((df_copy, pred_))
    train_ = df_[:num_train]
    test_ = df_[num_train:]

    df_volume_ = np.vstack((train_, test_))
    print(df_volume_.shape[0])
    inputs_ = df_volume_[df_volume_.shape[0] - test_.shape[0] - window:]
    inputs_ = inputs_.reshape(-1, 1)
    inputs_ = sc.transform(inputs_)

    X_test_2 = []

    for k in range(window, num_2):
        X_test_3 = np.reshape(inputs_[k - window:k, 0], (window, 1))
        X_test_2.append(X_test_3)

    X_test_ = np.stack(X_test_2)
    predict_ = model_LSTM.predict(X_test_)
    pred_ = sc.inverse_transform(predict_)
    prediction_full.append(pred_[-1][0])
    df_copy = df_[j:]

prediction_full_new = np.vstack((predict, np.array(prediction_full).reshape(-1, 1)))

df_date = data[['Year']]

for h in range(5):
    df_date_add = pd.to_datetime(df_date['Year'].iloc[-1]) + pd.DateOffset(months=1)
    df_date_add = pd.DataFrame([df_date_add.strftime("%m-%Y")], columns=['Year'])
    df_date = df_date.append(df_date_add)
df_date = df_date.reset_index(drop=True)

print(df_date)


plt.figure(figsize=(20, 7))
plt.plot(data['Year'].values, data['Price'], color='red', label='Real price')
plt.plot(df_date['Year'][-prediction_full_new.shape[0]:].values, prediction_full_new, color='blue', label='Predicted price')
plt.xticks(np.arange(10, data.shape[0], 20))
plt.title('Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
