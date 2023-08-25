import torch

W = torch.ones((2,), requires_grad=True)

with torch.no_grad():
  W = W + 2

print(W)
