# https://www.python-engineer.com/courses/pytorchbeginner/03-autograd/
import torch

# requires_grad = True -> tracks all operations on the tensor.
w = torch.ones(3, requires_grad=True)
print(w)  # >>> tensor([1., 1., 1.], requires_grad=True)
# x is created by the user -> grad_fn is None
print(w.grad_fn)  # >>> None, None

c = torch.tensor([2])
# x was created as a result of an operation, so it has a grad_fn attribute.
x = w + c
print(x)  # >>> tensor([3., 3., 3.], grad_fn=<AddBackward0>)
# grad_fn: references a Function that has created the Tensor
print(x.grad, x.grad_fn)  # >>> None, <AddBackward0 object at 0x11b082ef0>
#x.retain_grad()

# Do more operations on x
y = x * 3
print(y)  # >>> tensor([27., 27., 27.], grad_fn=<MulBackward0>)
print(y.grad_fn)  # >>> None, <AddBackward0 object at 0x11b082ef0>
#y.retain_grad()

# Make the output scalar
z = y.mean()
print(z)  # >>> tensor(27., grad_fn=<MeanBackward0>)
print(z.shape)  # >>> torch.Size([])
print(z.grad_fn)  # >>> None, <MeanBackward0 object at 0x11b082ef0>
#z.retain_grad()

print("#" * 50, 1)

z.backward()

# print("z.grad ->", z.grad)
# print("y.grad ->", y.grad)
# print("x.grad ->", x.grad)
print("w.grad ->", w.grad)
