# https://www.python-engineer.com/courses/pytorchbeginner/03-autograd/
import torch

# requires_grad = True -> tracks all operations on the tensor.
x = torch.ones(3, requires_grad=True)
y = x + 2

# y was created as a result of an operation, so it has a grad_fn attribute.
print(x)          # >>> tensor([1., 1., 1.], requires_grad=True)
print(x.grad_fn)  # >>> None, x is created by the user -> grad_fn is None
print(y)          # >>> tensor([3., 3., 3.], grad_fn=<AddBackward0>)
print(y.grad_fn)  # >>> <AddBackward0 object at 0x11b082ef0>, grad_fn: references a Function that has created the Tensor

# Do more operations on y
z = y * y * 3
print(z)          # >>> tensor([27., 27., 27.], grad_fn=<MulBackward0>)

# Make the output scalar
z = z.mean()
print(z)          # >>> tensor(27., grad_fn=<MeanBackward0>)
print(z.shape)    # >>> torch.Size([])

print("#" * 50, 1)

z.backward()
print(x.grad)  # dz/dx

print("#" * 50, 2)

a = torch.randn(2, 2)
print(a.requires_grad)

b = ((a * 3) / (a - 1))
print(b.grad_fn)

a.requires_grad_(True)

print(a.requires_grad)

c = (a * a).sum()
print(c.grad_fn)

print("#" * 50, 3)

a = torch.randn(2, 2, requires_grad=True)
print(a.requires_grad)
b = a.detach()
print(b.requires_grad)
print(b is a)

print("#" * 50, 4)

a = torch.randn(2, 2, requires_grad=True)
print(a.requires_grad)
with torch.no_grad():
    print(a.requires_grad)
    b = a * 2
    print(b.requires_grad)

print("#" * 50, 5)

weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    # just a dummy example
    model_output = (weights * 3).sum()
    print("[model_output {0}]:".format(epoch), model_output)
    model_output.backward()

    print("weights.grad:", weights.grad)

    # optimize model, i.e. adjust weights...
    with torch.no_grad():
        weights -= 0.1 * weights.grad

    # this is important! It affects the final weights & output
    weights.grad.zero_()

print("weights:", weights)

model_output = (weights * 3).sum()
print("[model_output final]:", model_output)

print("#" * 50, 6)

