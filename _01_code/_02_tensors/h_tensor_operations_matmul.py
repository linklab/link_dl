import torch

# vector x vector: dot product
t1 = torch.randn(3)
t2 = torch.randn(3)
print(torch.matmul(t1, t2).size())  # torch.Size([])

# matrix x vector: broadcasted dot
t3 = torch.randn(3, 4)
t4 = torch.randn(4)
print(torch.matmul(t3, t4).size())  # torch.Size([3])

# batched matrix x vector: broadcasted dot
t5 = torch.randn(10, 3, 4)
t6 = torch.randn(4)
print(torch.matmul(t5, t6).size())  # torch.Size([10, 3])

# batched matrix x batched matrix: bmm
t7 = torch.randn(10, 3, 4)
t8 = torch.randn(10, 4, 5)
print(torch.matmul(t7, t8).size())  # torch.Size([10, 3, 5])

# batched matrix x matrix: bmm
t9 = torch.randn(10, 3, 4)
t10 = torch.randn(4, 5)
print(torch.matmul(t9, t10).size())  # torch.Size([10, 3, 5])
