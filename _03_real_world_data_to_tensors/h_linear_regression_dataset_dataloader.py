import torch
from torch.utils.data import Dataset, DataLoader


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

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        y = torch.sum(self.x[idx] * self.m) + self.b + self.noise[idx]
        return {'input': self.x[idx], 'target': y}


if __name__ == "__main__":
    linear_regression_dataset = LinearRegressionDataset()

    for idx, sample in enumerate(linear_regression_dataset):
        print("{0} - {1}: {2}".format(idx, sample['input'], sample['target']))

    dataloader = DataLoader(
        dataset=linear_regression_dataset,
        batch_size=4,
        shuffle=True
    )

    print()

    for idx, batch in enumerate(dataloader):
        print("{0} - {1}: {2}".format(idx, batch['input'], batch['target']))

