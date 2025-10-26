import torch
from torch import nn

lstm = nn.LSTM(input_size=3, hidden_size=4, num_layers=2, bidirectional=True)

for name, parameter in lstm.named_parameters():
  print(name, parameter.shape)

print("#" * 100, 1)

# sequence size (L): 6, input size (F): 3
input = torch.randn(6, 3)

output, (hidden_state, _) = lstm(input)

for idx, out in enumerate(output):
  print(idx, out)     # shape: torch.Size([4])

print()

for idx, hidden in enumerate(hidden_state):
  print(idx, hidden)  # shape: torch.Size([4])
