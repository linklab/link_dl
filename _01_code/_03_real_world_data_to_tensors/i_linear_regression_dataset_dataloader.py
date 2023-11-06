import torch
from torch.utils.data import Dataset, DataLoader, random_split


class LinearRegressionDataset(Dataset):
  def __init__(self, N=50, m=-3, b=2, *args, **kwargs):
    # N: number of samples, e.g. 50
    # m: slope
    # b: offset
    super().__init__(*args, **kwargs)

    self.x = torch.rand(N, 2)
    self.noise = torch.rand(N) * 0.2
    self.m = m
    self.b = b
    self.y = (torch.sum(self.x * self.m) + self.b + self.noise).unsqueeze(-1)

  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]

  def __str__(self):
    str = "Data Size: {0}, Input Shape: {1}, Target Shape: {2}".format(
      len(self.x), self.x.shape, self.y.shape
    )
    return str


if __name__ == "__main__":
  linear_regression_dataset = LinearRegressionDataset()

  print(linear_regression_dataset)

  print("#" * 50, 1)

  for idx, sample in enumerate(linear_regression_dataset):
    input, target = sample
    print("{0} - {1}: {2}".format(idx, input, target))

  train_dataset, validation_dataset, test_dataset = random_split(linear_regression_dataset, [0.7, 0.2, 0.1])

  print("#" * 50, 2)

  print(len(train_dataset), len(validation_dataset), len(test_dataset))

  print("#" * 50, 3)

  train_data_loader = DataLoader(
    dataset=train_dataset,
    batch_size=4,
    shuffle=True
  )

  for idx, batch in enumerate(train_data_loader):
    input, target = batch
    print("{0} - {1}: {2}".format(idx, input, target))
