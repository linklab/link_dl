import torch

t1 = torch.ones(size=(2, 3))
t2 = torch.ones(size=(2, 3))
t3 = torch.add(t1, t2)
t4 = t1 + t2
print(t3)
print(t4)

print("#" * 30)

t5 = torch.sub(t1, t2)
t6 = t1 - t2
print(t5)
print(t6)

print("#" * 30)

t7 = torch.mul(t1, t2)
t8 = t1 * t2
print(t7)
print(t8)

print("#" * 30)

t9 = torch.div(t1, t2)
t10 = t1 / t2
print(t9)
print(t10)
