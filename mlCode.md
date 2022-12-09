# 학습 코드

```python
import math

import matplotlib.pyplot as plt

import keras

import pandas as pd

import numpy as np

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers import *

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping

df=pd.read_csv("/content/rn_20221123192941.csv")

print(df.shape)

df.head(5)

#df = df.fillna(0)

#train set과 test set 나누는 기준
devideTrainSetTestSet = 2000

training_set = df.iloc[:devideTrainSetTestSet, 2:3].values

test_set = df.iloc[devideTrainSetTestSet:, 2:3].values

# 0~1 값으로 바꿈

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(training_set)

# x_train에 0~59 y_train에 60, x_train에 1~60 y_train에 61 이런식으로 돌림

X_train = []

y_train = []

#몇개 단위로 나누는지
unitDevide = 60

for i in range(unitDevide, devideTrainSetTestSet):

   X_train.append(training_set_scaled[i-unitDevide:i, 0])

   y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model = Sequential()

model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))

model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = True))

model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = True))

model.add(Dropout(0.2))

model.add(LSTM(units = 50))

model.add(Dropout(0.2))

model.add(Dense(units = 1))

model.compile(optimizer = "adam", loss = "mean_squared_error")

model.fit(X_train, y_train, epochs = 100, batch_size = 32)

dataset_train = df.iloc[:devideTrainSetTestSet, 2:3]

dataset_test = df.iloc[devideTrainSetTestSet:, 2:3]

dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)

inputs = dataset_total[len(dataset_total) - len(dataset_test) - unitDevide:].values

inputs = inputs.reshape(-1,1)

inputs = sc.transform(inputs)

X_test = []

inputLen = len(inputs)

for i in range(unitDevide, inputLen):

   X_test.append(inputs[i-unitDevide:i, 0])

X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print(X_test.shape)

predicted = model.predict(X_test)

predicted = sc.inverse_transform(predicted)

plt.plot(df.loc[devideTrainSetTestSet:, "Date"],dataset_test.values, color = "red", label = "measured Precipitation")

plt.plot(df.loc[devideTrainSetTestSet:, "Date"],predicted, color = "blue", label = "predicted Precipitation")

plt.xticks(np.arange(0,len(predicted),50))

plt.title("Precipitation forecast")

plt.xlabel("Time")

plt.ylabel("Precipitation")

plt.legend()

plt.show()

```
