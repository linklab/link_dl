import torch
from torch import nn


class MyFirstModel(nn.Module):
  def __init__(self, n_input, n_hidden_unit, n_output):
    super().__init__()

    # Inputs to 1st hidden linear layer
    self.hidden = nn.Linear(n_input, n_hidden_unit)
    self.sigmoid = nn.Sigmoid()

    # Output layer
    self.output = nn.Linear(n_hidden_unit, n_output)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    x = self.hidden(x)
    x = self.sigmoid(x)
    x = self.output(x)
    x = self.softmax(x)
    return x


class MySecondModel(nn.Module):
  def __init__(self, n_input, n_hidden_unit, n_output):
    super().__init__()

    self.model = nn.Sequential(
      nn.Linear(n_input, n_hidden_unit),
      nn.Sigmoid(),
      nn.Linear(n_hidden_unit, n_output),
      nn.Softmax(dim=1)
    )

  def forward(self, x):
    x = self.model(x)
    return x


class MyThirdModel(nn.Module):
  def __init__(self, n_input, n_hidden_unit, n_output):
    super().__init__()

    hidden = nn.Linear(n_input, n_hidden_unit)
    sigmoid = nn.Sigmoid()
    output = nn.Linear(n_hidden_unit, n_output)
    softmax = nn.Softmax(dim=1)

    modules = [hidden, sigmoid, output, softmax]
    self.module_list = nn.ModuleList(modules)

  def forward(self, x):
    for f in self.module_list:
      x = f(x)
    return x


if __name__ == "__main__":
  batch_input = torch.randn(4, 10)

  my_first_model = MyFirstModel(n_input=10, n_hidden_unit=20, n_output=3)
  for name, param in my_first_model.named_parameters():
    print(name, param.shape)

  first_output = my_first_model(batch_input)
  print(first_output)

  print("#" * 50, 1)

  my_second_model = MySecondModel(n_input=10, n_hidden_unit=20, n_output=3)
  for name, param in my_second_model.named_parameters():
    print(name, param.shape)

  second_output = my_second_model(batch_input)
  print(second_output)

  print("#" * 50, 2)

  my_third_model = MyThirdModel(n_input=10, n_hidden_unit=20, n_output=3)

  for name, param in my_third_model.named_parameters():
    print(name, param.shape)

  third_output = my_third_model(batch_input)
  print(third_output)