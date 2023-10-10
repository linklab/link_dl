import torch

a = torch.tensor(3)
b = torch.tensor([
    [[3],
     [4],
     [5]]
])
c = torch.tensor([
    [[[1], [3]]],
    [[[4], [5]]]
])

print(a.shape, a.ndim)

print(b.shape, b.ndim)

print(c.shape, c.ndim)

rnn_cell = torch.nn.RNNCell(input_size=3, hidden_size=4)
input_data = torch.randn(size=(10, 3))
ret = rnn_cell(input_data)
print(ret.shape, "!!!")