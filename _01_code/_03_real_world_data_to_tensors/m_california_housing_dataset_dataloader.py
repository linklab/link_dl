import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class CaliforniaHousingDataset(Dataset):
  def __init__(self):
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()
    data_mean = np.mean(housing.data, axis=0)
    data_var = np.var(housing.data, axis=0)
    self.data = torch.tensor((housing.data - data_mean) / np.sqrt(data_var), dtype=torch.float32)
    self.target = torch.tensor(housing.target, dtype=torch.float32).unsqueeze(dim=-1)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    sample_data = self.data[idx]
    sample_target = self.target[idx]
    return sample_data, sample_target

  def __str__(self):
    str = "Data Size: {0}, Input Shape: {1}, Target Shape: {2}".format(
      len(self.data), self.data.shape, self.target.shape
    )
    return str


if __name__ == "__main__":
  california_housing_dataset = CaliforniaHousingDataset()

  print(california_housing_dataset)

  print("#" * 50, 1)

  for idx, sample in enumerate(california_housing_dataset):
    input, target = sample
    print("{0} - {1}: {2}".format(idx, input.shape, target.shape))

  train_dataset, validation_dataset, test_dataset = random_split(california_housing_dataset, [0.7, 0.2, 0.1])

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
