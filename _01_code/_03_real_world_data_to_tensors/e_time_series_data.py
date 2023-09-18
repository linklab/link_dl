import os
import numpy as np
import torch

torch.set_printoptions(edgeitems=2, threshold=50, linewidth=75)

bikes_path = os.path.join(os.path.pardir, os.path.pardir, "_00_data", "e_time-series-bike-sharing-dataset", "hour-fixed.csv")

bikes_numpy = np.loadtxt(
  fname=bikes_path, dtype=np.float32, delimiter=",", skiprows=1,
  converters={
    1: lambda x: float(x[8:10])  # 2011-01-07 --> 07 --> 7.0
  }
)
bikes = torch.from_numpy(bikes_numpy)
print(bikes.shape)
print(bikes)

daily_bikes = bikes.view(-1, 24, bikes.shape[1])
print(daily_bikes.shape)  # >>> torch.Size([730, 24, 17])

daily_bikes_data = daily_bikes[:, :, :-1]
daily_bikes_target = daily_bikes[:, :, -1].unsqueeze(dim=-1)

print(daily_bikes_data.shape)
print(daily_bikes_target.shape)

print("#" * 50, 1)

first_day_data = daily_bikes_data[0]
print(first_day_data.shape)

# Whether situation: 1: clear, 2:mist, 3: light rain/snow, 4: heavy rain/snow
print(first_day_data[:, 9].long())
eye_matrix = torch.eye(4)
print(eye_matrix)

weather_onehot = eye_matrix[first_day_data[:, 9].long() - 1]
print(weather_onehot.shape)
print(weather_onehot)

first_day_data_torch = torch.cat(tensors=(first_day_data, weather_onehot), dim=1)
print(first_day_data_torch.shape)
print(first_day_data_torch)

print("#" * 50, 2)

day_data_torch_list = []
for daily_idx in range(daily_bikes_data.shape[0]):  # range(730)
  day = daily_bikes_data[daily_idx]  # day.shape: [24, 17]
  weather_onehot = eye_matrix[day[:, 9].long() - 1]
  day_data_torch = torch.cat(tensors=(day, weather_onehot), dim=1)  # day_torch.shape: [24, 20]
  day_data_torch_list.append(day_data_torch)

print(len(day_data_torch_list))
daily_bikes_data = torch.stack(day_data_torch_list, dim=0)
print(daily_bikes_data.shape)

print("#" * 50, 3)

print(daily_bikes_data[:, :, :9].shape, daily_bikes_data[:, :, 10:].shape)
daily_bikes_data = torch.cat(
  [daily_bikes_data[:, :, :9], daily_bikes_data[:, :, 10:]],
  dim=2
)
print(daily_bikes_data.shape)

temperatures = daily_bikes_data[:, :, 9]
daily_bikes_data[:, :, 9] = (daily_bikes_data[:, :, 9] - torch.mean(temperatures)) / torch.std(temperatures)

# daily_bikes_data = daily_bikes_data.transpose(1, 2)
print(daily_bikes_data.shape)  # >>> torch.Size([730, 17, 24])
