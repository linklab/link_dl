import torch
from torch import optim

x = torch.ones(3)
w = torch.ones(3, requires_grad=True)

y = torch.tensor([1.1, 1.1, 1.1])

EPOCHS = 1200
LEARNING_RATE = 1e-2
optimizer = optim.SGD([w], lr=LEARNING_RATE)


def sigmoid(x):
  return 1.0 / (1.0 + (-x).exp())


for t in range(EPOCHS):
  y_pred = sigmoid(x * w)
  loss = (y_pred - y).pow(2).mean()

  loss.backward()
  optimizer.step()
  optimizer.zero_grad()

  print(w)
