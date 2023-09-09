import torch

a = torch.tensor(3)
b = torch.tensor([
    [[3],
     [4],
     [5]]
])
c = torch.tensor([
    [[[1], [3]]],
    [[[4], [5]]]
])

print(a.shape, a.ndim)

print(b.shape, b.ndim)

print(c.shape, c.ndim)