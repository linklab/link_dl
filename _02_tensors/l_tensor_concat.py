import torch

t1 = torch.zeros([2, 1, 3])
t2 = torch.zeros([2, 3, 3])
t3 = torch.zeros([2, 2, 3])

t4 = torch.cat([t1, t2, t3], dim=1)
print(t4.shape)

print("#" * 50, 1)

t5 = torch.arange(0, 3)  # tensor([0, 1, 2])
t6 = torch.arange(3, 8)  # tensor([3, 4, 5, 6, 7])

t7 = torch.cat((t5, t6), dim=0)
print(t7.shape)  # >>> torch.Size([8])
print(t7)  # >>> tensor([0, 1, 2, 3, 4, 5, 6, 7])

print("#" * 50, 2)

t8 = torch.arange(0, 6).reshape(2, 3)  # torch.Size([2, 3])
t9 = torch.arange(6, 12).reshape(2, 3)  # torch.Size([2, 3])

# 2차원 텐서간 병합
t10 = torch.cat((t8, t9), dim=0)
print(t10.size())  # >>> torch.Size([4, 3])
print(t10)
# >>> tensor([[ 0,  1,  2],
#             [ 3,  4,  5],
#             [ 6,  7,  8],
#             [ 9, 10, 11]])

t11 = torch.cat((t8, t9), dim=1)
print(t11.size())  # >>>torch.Size([2, 6])
print(t11)
# >>> tensor([[ 0,  1,  2,  6,  7,  8],
#             [ 3,  4,  5,  9, 10, 11]])

print("#" * 50, 3)

t12 = torch.arange(0, 6).reshape(2, 3)  # torch.Size([2, 3])
t13 = torch.arange(6, 12).reshape(2, 3)  # torch.Size([2, 3])
t14 = torch.arange(12, 18).reshape(2, 3)  # torch.Size([2, 3])

t15 = torch.cat((t12, t13, t14), dim=0)
print(t15.size())  # >>> torch.Size([6, 3])
print(t15)
# >>> tensor([[ 0,  1,  2],
#             [ 3,  4,  5],
#             [ 6,  7,  8],
#             [ 9, 10, 11],
#             [12, 13, 14],
#             [15, 16, 17]])

t16 = torch.cat((t12, t13, t14), dim=1)
print(t16.size())  # >>> torch.Size([2, 9])
print(t16)
# >>> tensor([[ 0,  1,  2,  6,  7,  8, 12, 13, 14],
#             [ 3,  4,  5,  9, 10, 11, 15, 16, 17]])

print("#" * 50, 4)

t17 = torch.arange(0, 6).reshape(1, 2, 3)  # torch.Size([1, 2, 3])
t18 = torch.arange(6, 12).reshape(1, 2, 3)  # torch.Size([1, 2, 3])

t19 = torch.cat((t17, t18), dim=0)
print(t19.size())  # >>> torch.Size([2, 2, 3])
print(t19)
# >>> tensor([[[ 0,  1,  2],
#              [ 3,  4,  5]],
#             [[ 6,  7,  8],
#              [ 9, 10, 11]]])

t20 = torch.cat((t17, t18), dim=1)
print(t20.size())  # >>> torch.Size([1, 4, 3])
print(t20)
# >>> tensor([[[ 0,  1,  2],
#              [ 3,  4,  5],
#              [ 6,  7,  8],
#              [ 9, 10, 11]]])

t21 = torch.cat((t17, t18), dim=2)
print(t21.size())  # >>> torch.Size([1, 2, 6])
print(t21)
# >>> tensor([[[ 0,  1,  2,  6,  7,  8],
#              [ 3,  4,  5,  9, 10, 11]]])
