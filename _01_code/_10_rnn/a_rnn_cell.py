import torch
from torch import nn

rnn_cell = nn.RNNCell(input_size=3, hidden_size=4)
# sequence: 6, input: 3
input = torch.randn(6, 3)

# hidden: 4
hx = torch.randn(4)
output_seq = []
for i in range(6):
  hx = rnn_cell(input[i], hx)
  output_seq.append(hx)

for idx, output in enumerate(output_seq):
  print(idx, output)