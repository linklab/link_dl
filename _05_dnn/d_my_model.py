import torch
from torch import nn


class MyModel(nn.Module):
  def __init__(self, n_input, n_unit1, n_output):
    super().__init__()

    # Inputs to 1st hidden linear layer
    self.hidden = nn.Linear(n_input, n_unit1)
    self.sigmoid = nn.Sigmoid()

    # Output layer
    self.output = nn.Linear(n_unit1, n_output)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    x = self.hidden(x)
    x = self.sigmoid(x)
    x = self.output(x)
    x = self.softmax(x)


if __name__ == "__main__":
  my_linear = MyLinear(in_features=4, out_features=3)
  sample_input = torch.randn(4)
  output = my_linear(sample_input)

  print("input.shape:", sample_input.shape)
  print("my_linear.weight.shape:", my_linear.weight.shape)
  print("my_linear.bias.shape:", my_linear.bias.shape)
  print("output.shape:", output.shape)
  print(output)

  print("#" * 50)

  my_linear_2 = nn.Linear(in_features=4, out_features=3)
  output_2 = my_linear(sample_input)
  print(output_2)

  assert output.equal(output_2)