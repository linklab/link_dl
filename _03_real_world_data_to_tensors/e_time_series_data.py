import os
import numpy as np
import torch
torch.set_printoptions(edgeitems=2, threshold=50, linewidth=75)

bikes_path = os.path.join(os.path.pardir, "_00_data", "e_time-series-bike-sharing-dataset", "hour-fixed.csv")

bikes_numpy = np.loadtxt(
    fname=bikes_path, dtype=np.float32, delimiter=",", skiprows=1,
    converters={
        1: lambda x: float(x[8:10])  # 2011-01-07 --> 07 --> 7
    }
)
bikes = torch.from_numpy(bikes_numpy)
print(bikes.shape)
print(bikes)

daily_bikes = bikes.view(-1, 24, bikes.shape[1])
print(daily_bikes.shape)  # >>> torch.Size([730, 24, 17])

print("#" * 50, 1)

first_day = daily_bikes[0]
print(first_day.shape)

# Whether situation: 1: clear, 2:mist, 3: light rain/snow, 4: heavy rain/snow
print(first_day[:, 9].long())
eye_matrix = torch.eye(4)
print(eye_matrix)
weather_onehot = eye_matrix[first_day[:, 9].long() - 1]
print(weather_onehot.shape)
print(weather_onehot)

first_day_torch = torch.cat(tensors=(first_day, weather_onehot), dim=1)
print(first_day_torch.shape)
print(first_day_torch)

print("#" * 50, 2)

day_torch_list = []
for daily_idx in range(daily_bikes.shape[0]):   # range(730)
    day = daily_bikes[daily_idx]  # day.shape: [24, 17]
    weather_onehot = eye_matrix[day[:, 9].long() - 1]
    day_torch = torch.cat(tensors=(day, weather_onehot), dim=1) # day_torch.shape: [24, 21]
    day_torch_list.append(day_torch)

print(len(day_torch_list))
daily_bikes_torch = torch.stack(day_torch_list, dim=0)
print(daily_bikes_torch.shape)

print("#" * 50, 3)

print(daily_bikes_torch[:, :, :9].shape, daily_bikes_torch[:, :, 10:].shape)
daily_bikes_torch = torch.cat(
    [daily_bikes_torch[:, :, :9], daily_bikes_torch[:, :, 10:]],
    dim=2
)
print(daily_bikes_torch.shape)

temperatures = daily_bikes_torch[:, :, 9]
daily_bikes_torch[:, 9, :] = (daily_bikes_torch[:, 9, :] - torch.mean(temperatures)) / torch.std(temperatures)

daily_bikes_torch = daily_bikes_torch.transpose(1, 2)
print(daily_bikes_torch.shape)  # >>> torch.Size([730, 17, 24])
