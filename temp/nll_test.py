import torch
from torch import nn

log_softmax = nn.LogSoftmax(dim=1)
loss_fn = nn.NLLLoss()

# input to NLLLoss is of size N x C = 3 x 5
input = torch.randn(3, 5, requires_grad=True)
# each element in target must have 0 <= value < C
target = torch.tensor([1, 0, 4])
loss = loss_fn(log_softmax(input), target)
loss.backward()

# 2D loss example (used, for example, with image inputs)
N, C = 5, 4
loss_fn = nn.NLLLoss()
data = torch.randn(N, 16, 10, 10)
conv = nn.Conv2d(16, C, (3, 3))
log_softmax = nn.LogSoftmax(dim=1)
# output of conv forward is of shape [N, C, 8, 8]
output = log_softmax(conv(data))
# each element in target must have 0 <= value < C
target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
# input to NLLLoss is of size N x C x height (8) x width (8)
loss = loss_fn(output, target)
loss.backward()