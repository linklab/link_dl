import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class WineDataset(Dataset):
  def __init__(self):
    wine_path = os.path.join(os.path.pardir, os.path.pardir, "_00_data", "d_tabular-wine", "winequality-white.csv")
    wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";", skiprows=1)
    wineq = torch.from_numpy(wineq_numpy)

    data = wineq[:, :-1]  # Selects all rows and all columns except the last
    data_mean = torch.mean(data, dim=0)
    data_var = torch.var(data, dim=0)
    self.data = (data - data_mean) / torch.sqrt(data_var)

    target = wineq[:, -1].long()  # treat labels as an integer
    eye_matrix = torch.eye(10)
    self.target = eye_matrix[target]

    assert len(self.data) == len(self.target)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    wine_feature = self.data[idx]
    wine_target = self.target[idx]
    return wine_feature, wine_target

  def __str__(self):
    str = "Data Size: {0}, Input Shape: {1}, Target Shape: {2}".format(
      len(self.data), self.data.shape, self.target.shape
    )
    return str


if __name__ == "__main__":
  wine_dataset = WineDataset()

  print(wine_dataset)

  print("#" * 50, 1)

  for idx, sample in enumerate(wine_dataset):
    input, target = sample
    print("{0} - {1}: {2}".format(idx, input.shape, target.shape))

  train_dataset, validation_dataset, test_dataset = random_split(wine_dataset, [0.7, 0.2, 0.1])

  print("#" * 50, 2)

  print(len(train_dataset), len(validation_dataset), len(test_dataset))

  print("#" * 50, 3)

  train_data_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    drop_last=True
  )

  for idx, batch in enumerate(train_data_loader):
    input, target = batch
    print("{0} - {1}: {2}".format(idx, input.shape, target.shape))
