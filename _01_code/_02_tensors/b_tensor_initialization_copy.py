import torch
import numpy as np

l1 = [1, 2, 3]
t1 = torch.Tensor(l1)

l2 = [1, 2, 3]
t2 = torch.tensor(l2)

l3 = [1, 2, 3]
t3 = torch.as_tensor(l3)

l1[0] = 100
l2[0] = 100
l3[0] = 100

print(t1)
print(t2)
print(t3)

print("#" * 100)

l4 = np.array([1, 2, 3])
t4 = torch.Tensor(l4)

l5 = np.array([1, 2, 3])
t5 = torch.tensor(l5)

l6 = np.array([1, 2, 3])
t6 = torch.as_tensor(l6)

l4[0] = 100
l5[0] = 100
l6[0] = 100

print(t4)
print(t5)
print(t6)
