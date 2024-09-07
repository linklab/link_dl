import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent) # BASE_PATH: /Users/yhhan/git/link_dl
import sys
sys.path.append(BASE_PATH)


class HourlyBikesDataset(Dataset):
  def __init__(self, X, y):
    self.X = X
    self.y = y

    assert len(self.X) == len(self.y)

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    X = self.X[idx]
    y = self.y[idx]
    return X, y

  def __str__(self):
    str = "Data Size: {0}, Input Shape: {1}, Target Shape: {2}".format(
      len(self.X), self.X.shape, self.y.shape
    )
    return str


def get_hourly_bikes_data(sequence_size=24, validation_size=96, test_size=24, y_normalizer=100):
  bikes_path = os.path.join(BASE_PATH, "_00_data", "e_time-series-bike-sharing-dataset", "hour-fixed.csv")

  bikes_numpy = np.loadtxt(
    fname=bikes_path, dtype=np.float32, delimiter=",", skiprows=1,
    converters={
      1: lambda x: float(x[8:10])  # 2011-01-07 --> 07 --> 7
    }
  )
  bikes_data = torch.from_numpy(bikes_numpy).to(torch.float) # >>> torch.Size([17520, 17])
  bikes_target = bikes_data[:, -1].unsqueeze(dim=-1)  # 'cnt'
  bikes_data = bikes_data[:, :-1]  # >>> torch.Size([17520, 16])

  eye_matrix = torch.eye(4)

  data_torch_list = []
  for idx in range(bikes_data.shape[0]):  # range(730)
    hour_data = bikes_data[idx]  # day.shape: [24, 17]
    weather_onehot = eye_matrix[hour_data[9].long() - 1]
    concat_data_torch = torch.cat(tensors=(hour_data, weather_onehot), dim=-1)  # day_torch.shape: [24, 21]
    data_torch_list.append(concat_data_torch)

  bikes_data = torch.stack(data_torch_list, dim=0)
  bikes_data = torch.cat([bikes_data[:, 1:9], bikes_data[:, 10:]], dim=-1)
  print(bikes_data.shape, "!!!")  # >>> torch.Size([17520, 18])

  data_size = len(bikes_data) - sequence_size
  train_size = data_size - (validation_size + test_size)

  #################################################################################################

  row_cursor = 0

  X_train_list = []
  y_train_regression_list = []
  for idx in range(0, train_size):
    sequence_data = bikes_data[idx: idx + sequence_size]
    sequence_target = bikes_target[idx + sequence_size - 1]
    X_train_list.append(sequence_data)
    y_train_regression_list.append(sequence_target)
    row_cursor += 1

  X_train = torch.stack(X_train_list, dim=0).to(torch.float)
  y_train_regression = torch.tensor(y_train_regression_list, dtype=torch.float32) / y_normalizer

  m = X_train.mean(dim=0, keepdim=True)
  s = X_train.std(dim=0, keepdim=True)
  X_train = (X_train - m) / s

  #################################################################################################

  X_validation_list = []
  y_validation_regression_list = []
  for idx in range(row_cursor, row_cursor + validation_size):
    sequence_data = bikes_data[idx: idx + sequence_size]
    sequence_target = bikes_target[idx + sequence_size - 1]
    X_validation_list.append(sequence_data)
    y_validation_regression_list.append(sequence_target)
    row_cursor += 1

  X_validation = torch.stack(X_validation_list, dim=0).to(torch.float)
  y_validation_regression = torch.tensor(y_validation_regression_list, dtype=torch.float32) / y_normalizer

  X_validation -= m
  X_validation /= s
  #################################################################################################

  X_test_list = []
  y_test_regression_list = []
  for idx in range(row_cursor, row_cursor + test_size):
    sequence_data = bikes_data[idx: idx + sequence_size]
    sequence_target = bikes_target[idx + sequence_size - 1]
    X_test_list.append(sequence_data)
    y_test_regression_list.append(sequence_target)
    row_cursor += 1

  X_test = torch.stack(X_test_list, dim=0).to(torch.float)
  y_test_regression = torch.tensor(y_test_regression_list, dtype=torch.float32) / y_normalizer

  X_test -= m
  X_test /= s

  return (
    X_train, X_validation, X_test,
    y_train_regression, y_validation_regression, y_test_regression
  )


if __name__ == "__main__":
  X_train, X_validation, X_test, y_train, y_validation, y_test = get_hourly_bikes_data(
    sequence_size=24, validation_size=96, test_size=24, y_normalizer=100
  )

  print("Train: {0}, Validation: {1}, Test: {2}".format(len(X_train), len(X_validation), len(X_test)))

  train_hourly_bikes_dataset = HourlyBikesDataset(X=X_train, y=y_train)
  validation_hourly_bikes_dataset = HourlyBikesDataset(X=X_validation, y=y_validation)
  test_houly_bikes_dataset = HourlyBikesDataset(X=X_test, y=y_test)

  train_data_loader = DataLoader(
    dataset=train_hourly_bikes_dataset, batch_size=32, shuffle=True, drop_last=True
  )

  # for idx, batch in enumerate(train_data_loader):
  #   input, target = batch
  #   print("{0} - {1}: {2}, {3}".format(idx, input.shape, target.shape, target))