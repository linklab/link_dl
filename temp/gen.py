import torch
from torch.utils.data import random_split

generator1 = torch.Generator().manual_seed(42)

train_data, test_data = random_split(range(10), [7, 3], generator=generator1)
print(list(train_data), list(test_data))
# >>> [2, 6, 1, 8, 4, 5, 0] [9, 3, 7]

train_data, test_data = random_split(range(10), [7, 3], generator=generator1)
print(list(train_data), list(test_data))
# >>> [2, 6, 1, 8, 4, 5, 0] [9, 3, 7]
