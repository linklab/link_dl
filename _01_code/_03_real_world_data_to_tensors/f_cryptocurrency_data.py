# https://finance.yahoo.com/quote/BTC-KRW/history/
import pandas as pd
from pathlib import Path
import os
import torch
import matplotlib.pyplot as plt

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent) # BASE_PATH: /Users/yhhan/git/link_dl
import sys
sys.path.append(BASE_PATH)


btc_krw_path = os.path.join(BASE_PATH, "_00_data", "k_cryptocurrency", "BTC_KRW.csv")
df = pd.read_csv(btc_krw_path)
print(df)

row_size = len(df)
print("row_size:", row_size)

columns = df.columns  #['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
print([column for column in columns])
date_list = df['Date']
df = df.drop(columns=['Date'])

print(df)
print("#" * 100, 0)

#################################################################################################

sequence_size = 10
validation_size = 100
test_size = 50

data_size = row_size - sequence_size + 1
print("data_size: {0}".format(data_size))
train_size = data_size - (validation_size + test_size)
print("train_size: {0}, validation_size: {1}, test_size: {2}".format(train_size, validation_size, test_size))

print("#" * 100, 1)

#################################################################################################

row_cursor = 0
y_normalizer = 1.0e7

X_train_list = []
y_train_regression_list = []
y_train_classification_list = []
y_train_date = []
for idx in range(0, train_size):
  sequence_data = df.iloc[idx: idx + sequence_size].values  # sequence_data.shape: (sequence_size, 5)
  X_train_list.append(torch.from_numpy(sequence_data))
  y_train_regression_list.append(df.iloc[idx + sequence_size - 1]["Close"])
  y_train_classification_list.append(
    1 if df.iloc[idx + sequence_size - 1]["Close"] >= df.iloc[idx + sequence_size - 2]["Close"] else 0
  )
  y_train_date.append(date_list[idx + sequence_size - 1])
  row_cursor += 1

X_train = torch.stack(X_train_list, dim=0).to(torch.float)
y_train_regression = torch.tensor(y_train_regression_list, dtype=torch.float32) / y_normalizer
y_train_classification = torch.tensor(y_train_classification_list, dtype=torch.int64)
print(y_train_classification)

m = X_train.mean(dim=0, keepdim=True)
s = X_train.std(dim=0, keepdim=True)
X_train -= m
X_train /= s
print(X_train.shape, y_train_regression.shape, y_train_classification.shape)
print("Label - Start Date: {0} ~ End Date: {1}".format(y_train_date[0], y_train_date[-1]))

print("#" * 100, 2)

#################################################################################################

X_validation_list = []
y_validation_regression_list = []
y_validation_classification_list = []
y_validation_date = []
for idx in range(row_cursor, row_cursor + validation_size):
  sequence_data = df.iloc[idx: idx + sequence_size].values     # sequence_data.shape: (sequence_size, 5)
  X_validation_list.append(torch.from_numpy(sequence_data))
  y_validation_regression_list.append(df.iloc[idx + sequence_size - 1]["Close"])
  y_validation_classification_list.append(
    1 if df.iloc[idx + sequence_size - 1]["Close"] >= df.iloc[idx + sequence_size - 2]["Close"] else 0
  )
  y_validation_date.append(date_list[idx + sequence_size - 1])
  row_cursor += 1

X_validation = torch.stack(X_validation_list, dim=0).to(torch.float)
y_validation_regression = torch.tensor(y_validation_regression_list, dtype=torch.float32) / y_normalizer
y_validation_classification = torch.tensor(y_validation_classification_list, dtype=torch.int64)
print(y_validation_classification)

X_validation = (X_validation - m) / s
print(X_validation.shape, y_validation_regression.shape, y_validation_classification.shape)
print("Label - Start Date: {0} ~ End Date: {1}".format(y_validation_date[0], y_validation_date[-1]))

print("#" * 100, 3)

#################################################################################################

X_test_list = []
y_test_regression_list = []
y_test_classification_list = []
y_test_date = []
for idx in range(row_cursor, row_cursor + test_size):
  sequence_data = df.iloc[idx: idx + sequence_size].values   # sequence_data.shape: (sequence_size, 5)
  X_test_list.append(torch.from_numpy(sequence_data))
  y_test_regression_list.append(df.iloc[idx + sequence_size - 1]["Close"])
  y_test_classification_list.append(
    1 if df.iloc[idx + sequence_size - 1]["Close"] > df.iloc[idx + sequence_size - 2]["Close"] else 0
  )
  y_test_date.append(date_list[idx + sequence_size - 1])
  row_cursor += 1

X_test = torch.stack(X_test_list, dim=0).to(torch.float)
y_test_regression = torch.tensor(y_test_regression_list, dtype=torch.float32) / y_normalizer
y_test_classification = torch.tensor(y_test_classification_list, dtype=torch.int64)
print(y_test_classification)
X_test = (X_test - m) / s
print(X_test.shape, y_test_regression.shape, y_test_classification.shape)
print("Label - Start Date: {0} ~ End Date: {1}".format(y_test_date[0], y_test_date[-1]))

#######################################################################################

fig, ax = plt.subplots(1, figsize=(13, 7))
ax.plot(y_train_date, y_train_regression * y_normalizer, label="y_train_regression", linewidth=2)
ax.plot(y_validation_date, y_validation_regression * y_normalizer, label="y_validation", linewidth=2)
ax.plot(y_test_date, y_test_regression * y_normalizer, label="y_test", linewidth=2)
ax.set_ylabel('Bitcoin [KRW]', fontsize=14)
ax.set_xticks(ax.get_xticks()[::200])
plt.ticklabel_format(style='plain', axis='y')
plt.xticks(rotation=25)
ax.legend(loc='upper left', fontsize=16)
plt.show()

