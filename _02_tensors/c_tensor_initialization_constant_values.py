import torch

t1 = torch.ones(size=(5,))  # or torch.ones(5)
t1_like = torch.ones_like(input=t1)
print(t1)  # >>> tensor([1., 1., 1., 1., 1.])
print(t1_like)  # >>> tensor([1., 1., 1., 1., 1.])

t2 = torch.zeros(size=(6,))  # or torch.zeros(6)
t2_like = torch.zeros_like(input=t2)
print(t2)  # >>> tensor([0., 0., 0., 0., 0., 0.])
print(t2_like)  # >>> tensor([0., 0., 0., 0., 0., 0.])

t3 = torch.empty(size=(4,))  # or torch.zeros(4)
t3_like = torch.empty_like(input=t3)
print(t3)  # >>> tensor([0., 0., 0., 0.])
print(t3_like)  # >>> tensor([0., 0., 0., 0.])

t4 = torch.eye(n=3)
print(t4)
