import torch

# torch.Tensor class
t1 = torch.Tensor([1, 2, 3], device='cpu')
print(t1.dtype)   # >>> torch.float32
print(t1.device)  # >>> cpu
print(t1.requires_grad)  # >>> False
print(t1.size())  # torch.Size([3])
print(t1.shape)   # torch.Size([3])

# if you have gpu device
# t1_cuda = t1.to(torch.device('cuda'))
# or you can use shorthand
# t1_cuda = t1.cuda()
t1_cpu = t1.cpu()

print("#" * 50, 1)

# torch.tensor function
t2 = torch.tensor([1, 2, 3], device='cpu')
print(t2.dtype)  # >>> torch.int64
print(t2.device)  # >>> cpu
print(t2.requires_grad)  # >>> False
print(t2.size())  # torch.Size([3])
print(t2.shape)  # torch.Size([3])

# if you have gpu device
# t2_cuda = t2.to(torch.device('cuda'))
# or you can use shorthand
# t2_cuda = t2.cuda()
t2_cpu = t2.cpu()

print("#" * 50, 2)

a1 = torch.tensor(1)			     # shape: torch.Size([]), ndims(=rank): 0
print(a1.shape, a1.ndim)

a2 = torch.tensor([1])		  	     # shape: torch.Size([1]), ndims(=rank): 1
print(a2.shape, a2.ndim)

a3 = torch.tensor([1, 2, 3, 4, 5])   # shape: torch.Size([5]), ndims(=rank): 1
print(a3.shape, a3.ndim)

a4 = torch.tensor([[1], [2], [3], [4], [5]])   # shape: torch.Size([5, 1]), ndims(=rank): 2
print(a4.shape, a4.ndim)

a5 = torch.tensor([                 # shape: torch.Size([3, 2]), ndims(=rank): 2
    [1, 2],
    [3, 4],
    [5, 6]
])
print(a5.shape, a5.ndim)

a6 = torch.tensor([                 # shape: torch.Size([3, 2, 1]), ndims(=rank): 3
    [[1], [2]],
    [[3], [4]],
    [[5], [6]]
])
print(a6.shape, a6.ndim)

a7 = torch.tensor([                 # shape: torch.Size([3, 1, 2, 1]), ndims(=rank): 4
    [[[1], [2]]],
    [[[3], [4]]],
    [[[5], [6]]]
])
print(a7.shape, a7.ndim)

a8 = torch.tensor([                 # shape: torch.Size([3, 1, 2, 3]), ndims(=rank): 4
    [[[1, 2, 3], [2, 3, 4]]],
    [[[3, 1, 1], [4, 4, 5]]],
    [[[5, 6, 2], [6, 3, 1]]]
])
print(a8.shape, a8.ndim)


a9 = torch.tensor([                 # shape: torch.Size([3, 1, 2, 3, 1]), ndims(=rank): 5
    [[[[1], [2], [3]], [[2], [3], [4]]]],
    [[[[3], [1], [1]], [[4], [4], [5]]]],
    [[[[5], [6], [2]], [[6], [3], [1]]]]
])
print(a9.shape, a9.ndim)

a10 = torch.tensor([                 # shape: torch.Size([4, 5]), ndims(=rank): 2
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],
])
print(a10.shape, a10.ndim)

a10 = torch.tensor([                 # shape: torch.Size([4, 1, 5]), ndims(=rank): 3
    [[1, 2, 3, 4, 5]],
    [[1, 2, 3, 4, 5]],
    [[1, 2, 3, 4, 5]],
    [[1, 2, 3, 4, 5]],
])
print(a10.shape, a10.ndim)

a11 = torch.tensor([                 # ValueError: expected sequence of length 3 at dim 3 (got 2)
    [[[1, 2, 3], [4, 5]]],
    [[[1, 2, 3], [4, 5]]],
    [[[1, 2, 3], [4, 5]]],
    [[[1, 2, 3], [4, 5]]],
])

