import torch
from torch.utils.data import random_split

generator1 = torch.Generator().manual_seed(42)
generator2 = torch.Generator().manual_seed(21)

train_data, test_data = random_split(range(10), [7, 3], generator=generator1)
print(list(train_data), list(test_data))

train_data, validation_data, test_data = random_split(range(30), [0.7, 0.2, 0.1], generator=generator2)
print(len(train_data), len(validation_data), len(test_data))