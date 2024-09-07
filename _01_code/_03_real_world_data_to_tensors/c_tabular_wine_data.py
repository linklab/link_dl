import csv
import os
import numpy as np

wine_path = os.path.join(os.path.pardir, os.path.pardir, "_00_data", "d_tabular-wine", "winequality-white.csv")
wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";", skiprows=1)
print(wineq_numpy.dtype)
print(wineq_numpy.shape)
print(wineq_numpy)
print()

col_list = next(csv.reader(open(wine_path), delimiter=';'))
print(col_list)
print()

print("#" * 50, 1)

import torch

wineq = torch.from_numpy(wineq_numpy)
print(wineq.dtype)
print(wineq.shape)
print()

data = wineq[:, :-1]  # Selects all rows and all columns except the last
print(data.dtype)
print(data.shape)
print(data)
print()

target = wineq[:, -1]  # Selects all rows and the last column
print(target.dtype)
print(target.shape)
print(target)
print()

target = target.long()  # treat labels as an integer
print(target.dtype)
print(target.shape)
print(target)
print()

print("#" * 50, 2)

eye_matrix = torch.eye(10)
# We use the 'target' tensor as indices to extract the corresponding rows from the identity matrix
# It can generate the one-hot vectors for each element in the 'target' tensor
onehot_target = eye_matrix[target]

print(onehot_target.shape)  # >>> torch.Size([4898, 10])
print(onehot_target[0])
print(onehot_target[1])
print(onehot_target[-2])
print(onehot_target)

print("#" * 50, 3)

data_mean = torch.mean(data, dim=0)
data_var = torch.var(data, dim=0)
data = (data - data_mean) / torch.sqrt(data_var)
print(data)

print("#" * 50, 4)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, onehot_target, test_size=0.2)

print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)


def get_wine_data():
  wine_path = os.path.join(os.path.pardir, os.path.pardir, "_00_data", "d_tabular-wine", "winequality-white.csv")
  wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";", skiprows=1)

  wineq = torch.from_numpy(wineq_numpy)

  data = wineq[:, :-1]  # Selects all rows and all columns except the last
  target = wineq[:, -1].long()  # treat labels as an integer

  eye_matrix = torch.eye(10)
  onehot_target = eye_matrix[target]

  data_mean = torch.mean(data, dim=0)
  data_var = torch.var(data, dim=0)
  data = (data - data_mean) / torch.sqrt(data_var)

  X_train, X_valid, y_train, y_valid = train_test_split(data, onehot_target, test_size=0.2)

  return X_train, X_valid, y_train, y_valid