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

print()

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
