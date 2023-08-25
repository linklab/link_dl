import torch

t1 = torch.dot(
  torch.tensor([2, 3]), torch.tensor([2, 1])
)
print(t1, t1.size())

t2 = torch.randn(2, 3)
t3 = torch.randn(3, 2)
t4 = torch.mm(t2, t3)
print(t4, t4.size())

t5 = torch.randn(10, 3, 4)
t6 = torch.randn(10, 4, 5)
t7 = torch.bmm(t5, t6)
print(t7.size())
