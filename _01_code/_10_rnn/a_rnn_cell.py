import torch
from torch import nn

rnn_cell = nn.RNNCell(input_size=3, hidden_size=4)

for name, parameter in rnn_cell.named_parameters():
  print(name, parameter.shape)

print("#" * 100)

# sequence size (L): 6, input size (F): 3
input = torch.randn(6, 3)

# hidden size: 4
hx = torch.randn(4)
output = []
for i in range(6):  # sequence size (L): 6
  hx = rnn_cell(input=input[i], hx=hx)
  output.append(hx)

for idx, out in enumerate(output):
  print(idx, out)



