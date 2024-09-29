import torch
from torch import nn
import torch.nn.functional as F

class ClassificationNet(nn.Module):
  def __init__(self, input_size, output_size):
    super(ClassificationNet, self).__init__()
    self.hid1 = nn.Linear(input_size, 3)
    self.relu1 = nn.ReLU()
    self.hid2 = nn.Linear(3, 3)
    self.relu2 = nn.ReLU()
    self.outp = nn.Linear(3, output_size)

  def forward(self, x):
    z = self.relu1(self.hid1(x))
    z = self.relu2(self.hid2(z))
    z = self.outp(z) # logits output
    return z

model = ClassificationNet(input_size=4, output_size=10)
x = torch.rand(size=(256, 4))
y = torch.empty(256, dtype=torch.long).random_(10)
z = model(x)	              # mu = 𝑁𝑁(𝑥_𝑖;𝜽)
ce_loss = nn.CrossEntropyLoss()
loss = ce_loss(z, y)
print(loss)
