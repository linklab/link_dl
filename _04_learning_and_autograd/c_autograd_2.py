# https://www.python-engineer.com/courses/pytorchbeginner/03-autograd/
import torch

a = torch.randn(2, 2)
print(a.requires_grad)

b = ((a * 3) / (a - 1))
print(b.grad_fn)

# b.backward()

a.requires_grad_(True)

print(a.requires_grad)

c = (a * a).sum()
print(c.grad_fn)

c.backward()

print(a.grad)

print("#" * 50, 1)

a = torch.randn(2, 2, requires_grad=True)
print(a.requires_grad)
b = a.detach()
print(b.requires_grad)
print(b is a)

from torchviz import make_dot

x = torch.ones(3, requires_grad=True)
y = 2 * x
z = 3 + x
r = (y + z).sum()
make_dot(r).render("torchviz_1", format="png")

# Detach
x = torch.ones(3, requires_grad=True)
y = 2 * x
z = 3 + x.detach()
r = (y + z).sum()
make_dot(r).render("torchviz_2", format="png")

print("#" * 50, 2)

a = torch.randn(2, 2, requires_grad=True)
b = a * 2
make_dot(b).render("torchviz_3", format="png")

a = torch.randn(2, 2, requires_grad=True)
print(a.requires_grad)
with torch.no_grad():
  print(a.requires_grad)
  b = a * 2

print(a.requires_grad)
print(b.requires_grad)
make_dot(b).render("torchviz_4", format="png")
