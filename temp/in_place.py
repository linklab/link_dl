import torch

W = torch.ones((2,), requires_grad=True)

with torch.no_grad():
  W += 2

print(W, W.requires_grad)

c = torch.rand((2, 3), dtype=torch.float64) * 20.
d = c.type(torch.int32)