import torch
from torch import nn

class MyLinear(nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    self.weight = nn.Parameter(
        torch.randn(out_features, in_features)
    )
    self.bias = nn.Parameter(
        torch.randn(out_features)
    )
  def forward(self, input):
      return torch.matmul(self.weight, input) + self.bias
#      return self.weight @ input + self.bias

if __name__ == "__main__":
  my_linear = MyLinear(in_features=4, out_features=3)
  sample_input = torch.randn(4)
  output = my_linear(sample_input)
  print("input.shape:", sample_input.shape)  # >> torch.Size([4])
  print(
    "my_linear.weight.shape:", my_linear.weight.shape
  )
  # >> torch.Size([3, 4])
  print("my_linear.bias.shape:", my_linear.bias.shape)
  # >> torch.Size([3])
  print("output.shape:", output.shape)
  # >> torch.Size([3])
  print(output)  # >> tensor([-1.4657, 0.3310, 2.3507], grad_fn=<AddBackward0>)
