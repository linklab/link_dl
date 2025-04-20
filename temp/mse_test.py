import torch
from torch import nn

class RegressionNet(nn.Module):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.hid1 = nn.Linear(4, 7)
    self.relu1 = nn.ReLU()
    self.hid2 = nn.Linear(7, 7)
    self.relu2 = nn.ReLU()
    self.outp = nn.Linear(7, 2)

  def forward(self, x):
    z = self.relu1(self.hid1(x))
    z = self.relu2(self.hid2(z))
    z = self.outp(z) # logits output
    return z

model = RegressionNet()
x = torch.rand(size=(256, 4)) # (256, 4)
y = torch.rand(size=(256, 2)) # (256, 2): value
mu = model(x)		   # mu = 𝑁𝑁(𝑥_𝑖;𝜽)
loss_func = nn.MSELoss()
loss = loss_func(mu, y)
print(loss)
