import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class BikesDataset(Dataset):
    def __init__(self):
        bikes_path = os.path.join(os.path.pardir, "_00_data", "e_time-series-bike-sharing-dataset", "hour-fixed.csv")

        bikes_numpy = np.loadtxt(
            fname=bikes_path, dtype=np.float32, delimiter=",", skiprows=1,
            converters={
                1: lambda x: float(x[8:10])  # 2011-01-07 --> 07 --> 7
            }
        )
        bikes = torch.from_numpy(bikes_numpy)

        daily_bikes = bikes.view(-1, 24, bikes.shape[1])  # daily_bikes.shape: torch.Size([730, 24, 17])
        daily_bikes_target = daily_bikes[:, :, -1].unsqueeze(dim=-1)

        daily_bikes_data = daily_bikes[:, :, :-1]
        eye_matrix = torch.eye(4)

        day_data_torch_list = []
        for daily_idx in range(daily_bikes_data.shape[0]):  # range(730)
            day = daily_bikes_data[daily_idx]  # day.shape: [24, 17]
            weather_onehot = eye_matrix[day[:, 9].long() - 1]
            day_data_torch = torch.cat(tensors=(day, weather_onehot), dim=1)  # day_torch.shape: [24, 21]
            day_data_torch_list.append(day_data_torch)

        daily_bikes_data = torch.stack(day_data_torch_list, dim=0)

        daily_bikes_data = torch.cat(
            [daily_bikes_data[:, :, :9], daily_bikes_data[:, :, 10:]],
            dim=2
        )

        temperatures = daily_bikes_data[:, :, 9]
        daily_bikes_data[:, :, 9] = (daily_bikes_data[:, :, 9] - torch.mean(temperatures)) / torch.std(temperatures)

        self.daily_bikes_data = daily_bikes_data.transpose(1, 2)
        self.daily_bikes_target = daily_bikes_target.transpose(1, 2)

        assert len(self.daily_bikes_data) == len(self.daily_bikes_target)

    def __len__(self):
        return len(self.daily_bikes_data)

    def __getitem__(self, idx):
        bike_feature = self.daily_bikes_data[idx]
        bike_target = self.daily_bikes_target[idx]
        return {'input': bike_feature, 'target': bike_target}


if __name__ == "__main__":
    bikes_dataset = BikesDataset()

    for idx, sample in enumerate(bikes_dataset):
        print("{0} - {1}: {2}".format(idx, sample['input'].shape, sample['target'].shape))

    dataloader = DataLoader(
        dataset=bikes_dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True
    )

    print()

    for idx, batch in enumerate(dataloader):
        print("{0} - {1}: {2}".format(idx, batch['input'].shape, batch['target'].shape))

