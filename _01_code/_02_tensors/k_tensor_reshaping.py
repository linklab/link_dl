import torch

t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
t2 = t1.view(3, 2)  # Shape becomes (3, 2)
t3 = t1.reshape(1, 6)  # Shape becomes (1, 6)
print(t2)
print(t3)

t4 = torch.arange(8).view(2, 4)  # Shape becomes (2, 4)
t5 = torch.arange(6).view(2, 3)  # Shape becomes (2, 3)
print(t4)
print(t5)

print("#" * 50, 1)

# Original tensor with shape (1, 3, 1)
t6 = torch.tensor([[[1], [2], [3]]])

# Remove all dimensions of size 1
t7 = t6.squeeze()  # Shape becomes (3,)

# Remove dimension at position 0
t8 = t6.squeeze(0)  # Shape becomes (3, 1)
print(t7)
print(t8)

print("#" * 50, 2)

# Original tensor with shape (3,)
t9 = torch.tensor([1, 2, 3])

# Add a new dimension at position 1
t10 = t9.unsqueeze(1)  # Shape becomes (3, 1)
print(t10)

t11 = torch.tensor(
  [[1, 2, 3],
   [4, 5, 6]]
)
t12 = t11.unsqueeze(1)  # Shape becomes (2, 1, 3)
print(t12, t12.shape)

print("#" * 50, 3)

# Original tensor with shape (2, 3)
t13 = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Flatten the tensor
t14 = t13.flatten()  # Shape becomes (6,)

print(t14)

# Original tensor with shape (2, 2, 2)
t15 = torch.tensor([[[1, 2],
                     [3, 4]],
                    [[5, 6],
                     [7, 8]]])
t16 = torch.flatten(t15)

t17 = torch.flatten(t15, start_dim=1)

print(t16)
print(t17)

print("#" * 50, 4)

t18 = torch.randn(2, 3, 5)
print(t18.shape)  # >>> torch.Size([2, 3, 5])
print(torch.permute(t18, (2, 0, 1)).size())  # >>> torch.Size([5, 2, 3])

# Original tensor with shape (2, 3)
t19 = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Permute the dimensions
t20 = torch.permute(t19, dims=(0, 1))  # Shape becomes (2, 3) still
t21 = torch.permute(t19, dims=(1, 0))  # Shape becomes (3, 2)
print(t20)
print(t21)

# Transpose the tensor
t22 = torch.transpose(t19, 0, 1)  # Shape becomes (3, 2)

print(t22)

t23 = torch.t(t19)  # Shape becomes (3, 2)

print(t23)
