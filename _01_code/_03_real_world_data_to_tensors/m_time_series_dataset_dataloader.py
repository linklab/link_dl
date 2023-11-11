import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent) # BASE_PATH: /Users/yhhan/git/link_dl
import sys
sys.path.append(BASE_PATH)


class BikesDataset(Dataset):
  def __init__(self, train=True, test_days=1):
    self.train = train
    self.test_days = test_days

    bikes_path = os.path.join(BASE_PATH, "_00_data", "e_time-series-bike-sharing-dataset", "hour-fixed.csv")

    bikes_numpy = np.loadtxt(
      fname=bikes_path, dtype=np.float32, delimiter=",", skiprows=1,
      converters={
        1: lambda x: float(x[8:10])  # 2011-01-07 --> 07 --> 7
      }
    )
    bikes = torch.from_numpy(bikes_numpy)

    daily_bikes = bikes.view(-1, 24, bikes.shape[1])  # daily_bikes.shape: torch.Size([730, 24, 17])
    self.daily_bikes_target = daily_bikes[:, :, -1].unsqueeze(dim=-1)

    self.daily_bikes_data = daily_bikes[:, :, :-1]
    eye_matrix = torch.eye(4)

    day_data_torch_list = []
    for daily_idx in range(self.daily_bikes_data.shape[0]):  # range(730)
      day = self.daily_bikes_data[daily_idx]  # day.shape: [24, 17]
      weather_onehot = eye_matrix[day[:, 9].long() - 1]
      day_data_torch = torch.cat(tensors=(day, weather_onehot), dim=1)  # day_torch.shape: [24, 21]
      day_data_torch_list.append(day_data_torch)

    self.daily_bikes_data = torch.stack(day_data_torch_list, dim=0)

    self.daily_bikes_data = torch.cat(
      [self.daily_bikes_data[:, :, :9], self.daily_bikes_data[:, :, 10:]], dim=2
    )

    total_length = len(self.daily_bikes_data)
    self.train_bikes_data = self.daily_bikes_data[:total_length - test_days]
    self.train_bikes_targets = self.daily_bikes_target[:total_length - test_days]
    train_temperatures = self.train_bikes_data[:, :, 9]
    train_temperatures_mean = torch.mean(train_temperatures)
    train_temperatures_std = torch.std(train_temperatures)
    self.train_bikes_data[:, :, 9] = \
      (self.train_bikes_data[:, :, 9] - torch.mean(train_temperatures_mean)) / torch.std(train_temperatures_std)

    assert len(self.train_bikes_data) == len(self.train_bikes_targets)

    self.test_bikes_data = self.daily_bikes_data[-test_days:]
    self.test_bikes_targets = self.daily_bikes_target[-test_days:]

    self.test_bikes_data[:, :, 9] = \
      (self.test_bikes_data[:, :, 9] - torch.mean(train_temperatures_mean)) / torch.std(train_temperatures_std)

    assert len(self.test_bikes_data) == len(self.test_bikes_targets)

  def __len__(self):
    return len(self.train_bikes_data) if self.train is True else len(self.test_bikes_data)

  def __getitem__(self, idx):
    bike_feature = self.train_bikes_data[idx] if self.train is True else self.test_bikes_data[idx]
    bike_target = self.train_bikes_targets[idx] if self.train is True else self.test_bikes_targets[idx]
    return bike_feature, bike_target

  def __str__(self):
    if self.train is True:
      str = "Data Size: {0}, Input Shape: {1}, Target Shape: {2}".format(
        len(self.train_bikes_data), self.train_bikes_data.shape, self.train_bikes_targets.shape
      )
    else:
      str = "Data Size: {0}, Input Shape: {1}, Target Shape: {2}".format(
        len(self.test_bikes_data), self.test_bikes_data.shape, self.test_bikes_targets.shape
      )
    return str


if __name__ == "__main__":
  train_bikes_dataset = BikesDataset(train=True, test_days=1)
  print(train_bikes_dataset)

  print("#" * 50, 1)

  train_dataset, validation_dataset = random_split(train_bikes_dataset, [0.8, 0.2])

  print("[TRAIN]")
  for idx, sample in enumerate(train_dataset):
    input, target = sample
    print("{0} - {1}: {2}".format(idx, input.shape, target.shape))

  train_data_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, drop_last=True)

  for idx, batch in enumerate(train_data_loader):
    input, target = batch
    print("{0} - {1}: {2}".format(idx, input.shape, target.shape))

  print("#" * 50, 2)

  print("[VALIDATION]")
  for idx, sample in enumerate(validation_dataset):
    input, target = sample
    print("{0} - {1}: {2}".format(idx, input.shape, target.shape))

  validation_data_loader = DataLoader(dataset=validation_dataset, batch_size=32)

  for idx, batch in enumerate(validation_data_loader):
    input, target = batch
    print("{0} - {1}: {2}".format(idx, input.shape, target.shape))

  print("#" * 50, 3)

  test_dataset = BikesDataset(train=False, test_days=1)
  print(test_dataset)

  print("[TEST]")
  for idx, sample in enumerate(test_dataset):
    input, target = sample
    print("{0} - {1}: {2}".format(idx, input.shape, target.shape))

  test_data_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset))

  for idx, batch in enumerate(test_data_loader):
    input, target = batch
    print("{0} - {1}: {2}".format(idx, input.shape, target.shape))