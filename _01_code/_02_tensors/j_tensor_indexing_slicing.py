import torch

x = torch.tensor(
  [[0, 1, 2, 3, 4],
   [5, 6, 7, 8, 9],
   [10, 11, 12, 13, 14]]
)

print(x[1])  # >>> tensor([5, 6, 7, 8, 9])
print(x[:, 1])  # >>> tensor([1, 6, 11])
print(x[1, 2])  # >>> tensor(7)
print(x[:, -1])  # >>> tensor([4, 9, 14)

print("#" * 50, 1)

print(x[1:])  # >>> tensor([[ 5,  6,  7,  8,  9], [10, 11, 12, 13, 14]])
print(x[1:, 3:])  # >>> tensor([[ 8,  9], [13, 14]])

print("#" * 50, 2)

y = torch.zeros((6, 6))
y[1:4, 2] = 1
print(y)

print(y[1:4, 1:4])

print("#" * 50, 3)

z = torch.tensor(
  [[1, 2, 3, 4],
   [2, 3, 4, 5],
   [5, 6, 7, 8]]
)
print(z[:2])
print(z[1:, 1:3])
print(z[:, 1:])

z[1:, 1:3] = 0
print(z)
