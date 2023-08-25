import torch

t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
t2 = torch.tensor([[7, 8, 9], [10, 11, 12]])

t3 = torch.stack([t1, t2], dim=0)
t4 = torch.cat([t1.unsqueeze(dim=0), t2.unsqueeze(dim=0)], dim=0)
print(t3.shape, t3.equal(t4))

t5 = torch.stack([t1, t2], dim=1)
t6 = torch.cat([t1.unsqueeze(dim=1), t2.unsqueeze(dim=1)], dim=1)
print(t5.shape, t5.equal(t6))

t7 = torch.stack([t1, t2], dim=2)
t8 = torch.cat([t1.unsqueeze(dim=2), t2.unsqueeze(dim=2)], dim=2)
print(t7.shape, t7.equal(t8))

print("#" * 50, 1)

t9 = torch.arange(0, 3)  # tensor([0, 1, 2])
t10 = torch.arange(3, 6)  # tensor([3, 4, 5])

print(t9.size(), t10.size())
# >>> torch.Size([3]) torch.Size([3])

t11 = torch.stack((t9, t10), dim=0)
print(t11.size())  # >>> torch.Size([2,3])
print(t11)
# >>> tensor([[0, 1, 2],
#             [3, 4, 5]])

t12 = torch.cat((t9.unsqueeze(0), t10.unsqueeze(0)), dim=0)
print(t11.equal(t12))
# >>> True

t13 = torch.stack((t9, t10), dim=1)
print(t13.size())  # >>> torch.Size([3,2])
print(t13)
# >>> tensor([[0, 3],
#             [1, 4],
#             [2, 5]])
t14 = torch.cat((t9.unsqueeze(1), t10.unsqueeze(1)), dim=1)
print(t13.equal(t14))
# >>> True
