import torch
from torch import nn

class RegressionNet(nn.Module):
  def __init__(self, input_size):
    super(RegressionNet, self).__init__()
    self.hid1 = nn.Linear(input_size, 3)
    self.relu1 = nn.ReLU()
    self.hid2 = nn.Linear(3, 3)
    self.relu2 = nn.ReLU()
    self.outp = nn.Linear(3, 1)

  def forward(self, x):
    z = self.relu1(self.hid1(x))
    z = self.relu2(self.hid2(z))
    z = self.outp(z) # logits output
    return z

model = RegressionNet(input_size=4)
x = torch.rand(size=(256, 4))  # (256, 4)
y = torch.rand(size=(256, 1))  # (256, 1): value
mu = model(x)	            # mu = 𝑁𝑁(𝑥_𝑖;𝜽)
loss_func = nn.MSELoss()
loss = loss_func(mu, y)
print(loss)
