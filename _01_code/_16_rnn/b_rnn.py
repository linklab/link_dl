import torch
from torch import nn


rnn1 = nn.RNN(input_size=3, hidden_size=4, num_layers=1)

for name, parameter in rnn1.named_parameters():
  print(name, parameter.shape)

# sequence size (L): 6, input size (F): 3
input = torch.randn(6, 3)

output, hidden_state = rnn1(input)

for idx, out in enumerate(output):
  print(idx, out)     # shape: torch.Size([4])

print()

for idx, hidden in enumerate(hidden_state):
  print(idx, hidden)  # shape: torch.Size([4])


print("#" * 100, 1)
##############################################################################

rnn2 = nn.RNN(input_size=3, hidden_size=4, num_layers=2)

for name, parameter in rnn2.named_parameters():
  print(name, parameter.shape)

print("#" * 100, 2)

# sequence size (L): 6, input size (F): 3
input = torch.randn(6, 3)

output, hidden_state = rnn2(input)

for idx, out in enumerate(output):
  print(idx, out)     # shape: torch.Size([4])

print()

for idx, hidden in enumerate(hidden_state):
  print(idx, hidden)  # shape: torch.Size([4])
