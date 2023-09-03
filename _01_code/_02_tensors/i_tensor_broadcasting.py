import torch

t1 = torch.tensor([1.0, 2.0, 3.0])
t2 = 2.0
print(t1 * t2)

print("#" * 50, 1)

t3 = torch.tensor([[0, 1], [2, 4], [10, 10]])
t4 = torch.tensor([4, 5])
print(t3 - t4)

print("#" * 50, 2)

t5 = torch.tensor([[1., 2.], [3., 4.]])
print(t5 + 2.0)  # t5.add(2.0)
print(t5 - 2.0)  # t5.sub(2.0)
print(t5 * 2.0)  # t5.mul(2.0)
print(t5 / 2.0)  # t5.div(2.0)

print("#" * 50, 3)


def normalize(x):
  return x / 255


t6 = torch.randn(3, 28, 28)
print(normalize(t6).size())

print("#" * 50, 4)

t7 = torch.tensor([[1, 2], [0, 3]])  # torch.Size([2, 2])
t8 = torch.tensor([[3, 1]])  # torch.Size([1, 2])
t9 = torch.tensor([[5], [2]])  # torch.Size([2, 1])
t10 = torch.tensor([7])  # torch.Size([1])
print(t7 + t8)   # >>> tensor([[4, 3], [3, 4]])
print(t7 + t9)   # >>> tensor([[6, 7], [2, 5]])
print(t8 + t9)   # >>> tensor([[8, 6], [5, 3]])
print(t7 + t10)  # >>> tensor([[ 8, 9], [ 7, 10]])

print("#" * 50, 5)

t11 = torch.ones(4, 3, 2)
t12 = t11 * torch.rand(3, 2)  # 3rd & 2nd dims identical to t11, dim 0 absent
print(t12.shape)

t13 = torch.ones(4, 3, 2)
t14 = t13 * torch.rand(3, 1)  # 3rd dim = 1, 2nd dim is identical to t13
print(t14.shape)

t15 = torch.ones(4, 3, 2)
t16 = t15 * torch.rand(1, 2)  # 3rd dim is identical to t15, 2nd dim is 1
print(t16.shape)

t17 = torch.ones(5, 3, 4, 1)
t18 = torch.rand(3, 1, 1)  # 2nd dim is identical to t17, 3rd and 4th dims are 1
print((t17 + t18).size())

print("#" * 50, 6)

t19 = torch.empty(5, 1, 4, 1)
t20 = torch.empty(3, 1, 1)
print((t19 + t20).size())  # torch.Size([5, 3, 4, 1])

t21 = torch.empty(1)
t22 = torch.empty(3, 1, 7)
print((t21 + t22).size())  # torch.Size([3, 1, 7])

t23 = torch.ones(3, 3, 3)
t24 = torch.ones(3, 1, 3)
print((t23 + t24).size())  # torch.Size([3, 3, 3])

# t25 = torch.empty(5, 2, 4, 1)
# t26 = torch.empty(3, 1, 1)
# print((t25 + t26).size())
# RuntimeError: The size of tensor a (2) must match
# the size of tensor b (3) at non-singleton dimension 1

print("#" * 50, 7)

t27 = torch.ones(4) * 5
print(t27)  # >>> tensor([ 5, 5, 5, 5])

t28 = torch.pow(t27, 2)
print(t28)  # >>> tensor([ 25, 25, 25, 25])

exp = torch.arange(1., 5.)  # tensor([ 1.,  2.,  3.,  4.])
a = torch.arange(1., 5.)  # tensor([ 1.,  2.,  3.,  4.])
t29 = torch.pow(a, exp)
print(t29)  # >>> tensor([   1.,    4.,   27.,  256.])
