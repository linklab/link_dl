import torch
from torch import nn


class MyLinear(nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    self.weight = nn.Parameter(torch.randn(out_features, in_features))
    self.bias = nn.Parameter(torch.randn(out_features))

  def forward(self, input):
    return input @ self.weight.T + self.bias + self.bias   # = torch.matmul(input, self.weight.T)


if __name__ == "__main__":
  my_linear = MyLinear(in_features=4, out_features=3)
  sample_input = torch.randn(4)
  output = my_linear(sample_input)

  print("input.shape:", sample_input.shape)
  print("my_linear.weight.shape:", my_linear.weight.shape)
  print("my_linear.bias.shape:", my_linear.bias.shape)
  print("output.shape:", output.shape)
  print(output)

  print("#" * 50, 1)

  my_linear_2 = nn.Linear(in_features=4, out_features=3)
  output_2 = my_linear_2(sample_input)
  print(output_2)

  assert output.shape == output_2.shape

  print("#" * 50, 2)

  batch_input = torch.randn(10, 4)
  batch_output = my_linear(batch_input)
  print(batch_output)

  print("#" * 50, 3)

  batch_input_2 = torch.randn(10, 3, 80, 80)
  # batch_output_2 = my_linear(batch_input_2)
  # print(batch_output.shape)