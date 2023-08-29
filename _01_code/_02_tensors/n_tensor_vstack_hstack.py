import torch

t1 = torch.tensor([1, 2, 3])
t2 = torch.tensor([4, 5, 6])
t3 = torch.vstack((t1, t2))
print(t3)
# >>> tensor([[1, 2, 3],
#             [4, 5, 6]])

t4 = torch.tensor([[1], [2], [3]])
t5 = torch.tensor([[4], [5], [6]])
t6 = torch.vstack((t4, t5))
# >>> tensor([[1],
#             [2],
#             [3],
#             [4],
#             [5],
#             [6]])

t7 = torch.tensor([
  [[1, 2, 3], [4, 5, 6]],
  [[7, 8, 9], [10, 11, 12]]
])
print(t7.shape)
# >>> (2, 2, 3)

t8 = torch.tensor([
  [[13, 14, 15], [16, 17, 18]],
  [[19, 20, 21], [22, 23, 24]]
])
print(t8.shape)
# >>> (2, 2, 3)

t9 = torch.vstack([t7, t8])
print(t9.shape)
# >>> (4, 2, 3)

print(t9)
# >>> tensor([[[ 1,  2,  3],
#              [ 4,  5,  6]],
#             [[ 7,  8,  9],
#              [10, 11, 12]],
#             [[13, 14, 15],
#              [16, 17, 18]],
#             [[19, 20, 21],
#              [22, 23, 24]]])

print("#" * 50, 1)

t10 = torch.tensor([1, 2, 3])
t11 = torch.tensor([4, 5, 6])
t12 = torch.hstack((t10, t11))
print(t12)
# >>> tensor([1, 2, 3, 4, 5, 6])

t13 = torch.tensor([[1], [2], [3]])
t14 = torch.tensor([[4], [5], [6]])
t15 = torch.hstack((t13, t14))
print(t15)
# >>> tensor([[1, 4],
#             [2, 5],
#             [3, 6]])

t16 = torch.tensor([
  [[1, 2, 3], [4, 5, 6]],
  [[7, 8, 9], [10, 11, 12]]
])
print(t16.shape)
# >>> (2, 2, 3)

t17 = torch.tensor([
  [[13, 14, 15], [16, 17, 18]],
  [[19, 20, 21], [22, 23, 24]]
])
print(t17.shape)
# >>> (2, 2, 3)

t18 = torch.hstack([t16, t17])
print(t18.shape)
# >>> (2, 4, 3)

print(t18)
# >>> tensor([[[ 1,  2,  3],
#              [ 4,  5,  6],
#              [13, 14, 15],
#              [16, 17, 18]],
#             [[ 7,  8,  9],
#              [10, 11, 12],
#              [19, 20, 21],
#              [22, 23, 24]]])
