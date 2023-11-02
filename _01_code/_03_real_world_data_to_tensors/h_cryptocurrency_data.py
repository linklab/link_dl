# https://finance.yahoo.com/quote/BTC-KRW/history/
import pandas as pd
from pathlib import Path
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent) # BASE_PATH: /Users/yhhan/git/link_dl
import sys
sys.path.append(BASE_PATH)


btc_krw_path = os.path.join(os.path.pardir, os.path.pardir, "_00_data", "k_cryptocurrency", "BTC_KRW.csv")
df = pd.read_csv(btc_krw_path)
print(df)

row_size = len(df)
print("row_size:", row_size)

columns = df.columns  #['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
print([column for column in columns])
date_list = df['Date']
print(date_list)
df = df.drop(columns=['Date'])

print("#" * 100, 0)

#################################################################################################

sequence_size = 10
validation_size = 100
test_size = 10

data_size = row_size - sequence_size
print("data_size: {0}".format(data_size))
train_size = data_size - (validation_size + test_size)
print("train_size: {0}, validation_size: {1}, test_size: {2}".format(train_size, validation_size, test_size))

print("#" * 100, 1)

#################################################################################################

row_cursor = 0
y_normalizer = 1.0e7

X_train_list = []
y_train_list = []
y_train_date = []
for idx in range(0, train_size):
  sequence_data = df.iloc[idx: idx + sequence_size].values
  sequence_data = sequence_data.astype(np.float32)  # sequence_data.shape: (sequence_size, 5)
  X_train_list.append(torch.from_numpy(sequence_data))
  y_train_list.append(df.iloc[idx + sequence_size]["Close"])
  y_train_date.append(date_list[idx + sequence_size])
  row_cursor += 1

X_train = torch.stack(X_train_list, dim=0)
y_train = torch.Tensor(y_train_list) / y_normalizer
m = X_train.mean(dim=0, keepdim=True)
s = X_train.std(dim=0, unbiased=False, keepdim=True)
X_train -= m
X_train /= s
print(X_train.shape, y_train.shape)
print("Label - Start Date: {0} ~ End Date: {1}".format(y_train_date[0], y_train_date[-1]))

print("#" * 100, 2)

#################################################################################################

X_validation_list = []
y_validation_list = []
y_validation_date = []
for idx in range(row_cursor, row_cursor + validation_size):
  sequence_data = df.iloc[idx: idx + sequence_size].values
  sequence_data = sequence_data.astype(np.float32)  # sequence_data.shape: (sequence_size, 5)
  X_validation_list.append(torch.from_numpy(sequence_data))
  y_validation_list.append(df.iloc[idx + sequence_size]["Close"])
  y_validation_date.append(date_list[idx + sequence_size])
  row_cursor += 1

X_validation = torch.stack(X_validation_list, dim=0)
y_validation = torch.Tensor(y_validation_list) / y_normalizer
X_validation -= m
X_validation /= s
print(X_validation.shape, y_validation.shape)
print("Label - Start Date: {0} ~ End Date: {1}".format(y_validation_date[0], y_validation_date[-1]))

print("#" * 100, 3)

#################################################################################################

X_test_list = []
y_test_list = []
y_test_date = []
for idx in range(row_cursor, row_cursor + test_size):
  sequence_data = df.iloc[idx: idx + sequence_size].values
  sequence_data = sequence_data.astype(np.float32)  # sequence_data.shape: (sequence_size, 5)
  X_test_list.append(torch.from_numpy(sequence_data))
  y_test_list.append(df.iloc[idx + sequence_size]["Close"])
  y_test_date.append(date_list[idx + sequence_size])
  row_cursor += 1

X_test = torch.stack(X_test_list, dim=0)
y_test = torch.Tensor(y_test_list) / y_normalizer
X_test -= m
X_test /= s
print(X_test.shape, y_test.shape)
print("Label - Start Date: {0} ~ End Date: {1}".format(y_test_date[0], y_test_date[-1]))

#######################################################################################

fig, ax = plt.subplots(1, figsize=(13, 7))
ax.plot(y_train_date, y_train * y_normalizer, label="y_train", linewidth=2)
ax.plot(y_validation_date, y_validation * y_normalizer, label="y_validation", linewidth=2)
ax.plot(y_test_date, y_test * y_normalizer, label="y_test", linewidth=2)
ax.set_ylabel('Bitcoin [KRW]', fontsize=14)
ax.set_xticks(ax.get_xticks()[::200])
plt.ticklabel_format(style='plain', axis='y')
plt.xticks(rotation=25)
ax.legend(loc='upper left', fontsize=16)
plt.show()

