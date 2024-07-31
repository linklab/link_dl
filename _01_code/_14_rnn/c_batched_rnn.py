import torch
from torch import nn

rnn = nn.RNN(input_size=3, hidden_size=4, num_layers=2)

# sequence size (L): 6, batch size (N): 10, input size (F): 3
batch_input = torch.randn(6, 10, 3)

batch_output, batch_hidden_state = rnn(batch_input)

print(batch_output.shape)
for idx, out in enumerate(batch_output):
  print(idx, out.shape)     # shape: torch.Size([10, 4])

print()

print(batch_hidden_state.shape)
for idx, hidden in enumerate(batch_hidden_state):
  print(idx, hidden.shape)  # shape: torch.Size([10, 4])

print("#" * 100, 1)

rnn = nn.RNN(input_size=3, hidden_size=4, num_layers=2, batch_first=True)

# batch size (N): 10, sequence size (L): 6, input size (F): 3
batch_input = torch.randn(10, 6, 3)

batch_output, batch_hidden_state = rnn(batch_input)

print(batch_output.shape)
for idx, out in enumerate(batch_output):
  print(idx, out.shape)     # shape: torch.Size([6, 4])

print()

print(batch_hidden_state.shape)
for idx, hidden in enumerate(batch_hidden_state):
  print(idx, hidden.shape)  # shape: torch.Size([10, 4])
