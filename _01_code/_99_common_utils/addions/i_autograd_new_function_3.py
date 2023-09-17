import torch
from torch import optim

x = torch.ones(3)
w = torch.ones(3, requires_grad=True)

y = torch.tensor([1.1, 1.1, 1.1])

EPOCHS = 1200
LEARNING_RATE = 1e-2
optimizer = optim.SGD([w], lr=LEARNING_RATE)


class Sigmoid(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x):
    y = 1.0 / (1.0 + (-x).exp())
    ctx.save_for_backward(y)
    return y

  @staticmethod
  def backward(ctx, grad_y):
    y = ctx.saved_tensors[0]
    grad_x = grad_y * y * (1 - y)
    return grad_x


def sigmoid(x):
  return Sigmoid.apply(x)


for t in range(EPOCHS):
  y_pred = sigmoid(x * w)
  loss = (y_pred - y).pow(2).mean()

  loss.backward()
  optimizer.step()
  optimizer.zero_grad()

  print(w)
