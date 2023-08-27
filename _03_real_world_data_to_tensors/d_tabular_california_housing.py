# https://medium.com/analytics-vidhya/implement-linear-regression-on-boston-housing-dataset-by-pytorch-c5d29546f938
# https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
import torch
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
print(housing.keys())

print(type(housing.data))
print(housing.data.dtype)
print(housing.data.shape)
print(housing.feature_names)

print(housing.target.shape)
print(housing.target_names)

print("#" * 50, 1)

import numpy as np

data_mean = np.mean(housing.data, axis=0)
data_var = np.var(housing.data, axis=0)
data = (housing.data - data_mean) / np.sqrt(data_var)
target = housing.target

print("#" * 50, 2)

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(data, target, test_size=0.2)

X_train = torch.from_numpy(X_train)
X_valid = torch.from_numpy(X_valid)
y_train = torch.from_numpy(y_train)
y_valid = torch.from_numpy(y_valid)

print(X_train.shape)
print(y_train.shape)

print(X_valid.shape)
print(y_valid.shape)
