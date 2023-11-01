import torch

x = torch.ones(3, requires_grad=True)
y = x.clone()
loss_1 = x - 2
loss_2 = y - 2
loss = loss_1 + loss_2
loss_sum = loss.sum()
loss_sum.backward()
print(x.grad)

from torchviz import make_dot
make_dot(loss_sum).render("torchviz_1", format="png")