import torch

# vector x vector
t1 = torch.randn(3)
t2 = torch.randn(3)
print(torch.matmul(t1, t2).size()) # torch.Size([])

# matrix x vector
t3 = torch.randn(3, 4)
t4 = torch.randn(4)
print(torch.matmul(t3, t4).size()) # torch.Size([3])

# batched matrix x broadcasted vector
t5 = torch.randn(10, 3, 4)
t6 = torch.randn(4)
print(torch.matmul(t5, t6).size()) # torch.Size([10, 3])

# batched matrix x batched matrix
t7 = torch.randn(10, 3, 4)
t8 = torch.randn(10, 4, 5)
print(torch.matmul(t7, t8).size()) # torch.Size([10, 3, 5])

# batched matrix x broadcasted matrix
t9 = torch.randn(10, 3, 4)
t10 = torch.randn(4, 5)
print(torch.matmul(t9, t10).size()) # torch.Size([10, 3, 5])