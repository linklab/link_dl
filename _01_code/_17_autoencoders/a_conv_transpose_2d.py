# https://cumulu-s.tistory.com/29
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
  def __init__(self):
    super(SimpleModel, self).__init__()
    self.model = nn.ConvTranspose2d(in_channels=1, out_channels=3, kernel_size=4, stride=1, padding=0)

  def forward(self, x):
    return self.model(x)


model = SimpleModel()

# Print model's original parameters.
for name, param in model.state_dict().items():
  print("\n[Parameter Name: {0}]".format(name))
  print("Param: ", param)
  print("Param shape: ", param.shape)

print("#" * 100, 0)

# sample values.
test_input = torch.Tensor([[[[1, 2, 3], [4, 5, 6]]]])
print("input size: ", test_input.shape)
print("test input: ", test_input)

print("#" * 100, 1)

result = model(test_input)

print("Result shape: ", result.shape)
print("Result: ", result)