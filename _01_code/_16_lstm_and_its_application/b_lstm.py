import torch
from torch import nn


lstm = nn.LSTM(input_size=3, hidden_size=4)

for name, parameter in lstm.named_parameters():
  print(name, parameter.shape)

print("#" * 100, 1)

# sequence size (L): 6, input size (F): 3
input = torch.randn(6, 3)

output, (hidden_state, cell_state) = lstm(input)

for idx, out in enumerate(output):
  print(idx, out)     # shape: torch.Size([4])

print()

for idx, (hidden, cell) in enumerate(zip(hidden_state, cell_state)):
  print(idx, hidden, cell)  # shape: [torch.Size([4]), torch.Size([4])]

print()
print("@" * 100)
print("@" * 100)
print("@" * 100)
print()

lstm = nn.LSTM(input_size=3, hidden_size=4, num_layers=2)

for name, parameter in lstm.named_parameters():
  print(name, parameter.shape)

print("#" * 100, 1)

# sequence size (L): 6, input size (F): 3
input = torch.randn(6, 3)

output, (hidden_state, cell_state) = lstm(input)

for idx, out in enumerate(output):
  print(idx, out)     # shape: torch.Size([4])

print()

for idx, (hidden, cell) in enumerate(zip(hidden_state, cell_state)):
  print(idx, hidden, cell)  # shape: [torch.Size([4]), torch.Size([4])]
