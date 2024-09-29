# https://www.python-engineer.com/courses/pytorchbeginner/03-autograd/
import torch

weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
  # just a dummy example
  loss = (weights * 3).sum()
  print("[loss {0}]:".format(epoch), loss)
  loss.backward()

  print("weights.grad:", weights.grad)

  # optimize model, i.e. adjust weights...
  with torch.no_grad():
    weights -= 0.1 * weights.grad
    # 'empty gradients' is important!
    # It affects the final weights & output
    weights.grad.zero_()

  print("weights:", weights)
  print()

loss = (weights * 3).sum()
print("\n[final loss]:", loss)

print("#" * 50, 1)

weights = torch.ones(4, requires_grad=True)
optimizer = torch.optim.SGD([weights], lr=0.1)

for epoch in range(3):
  # just a dummy example
  loss = (weights * 3).sum()
  print("[loss {0}]:".format(epoch), loss)
  loss.backward()

  print("weights.grad:", weights.grad)

  # optimize model, i.e. adjust weights & empty gradients
  optimizer.step()
  optimizer.zero_grad()

  print("weights:", weights)
  print()

loss = (weights * 3).sum()
print("\n[final loss]:", loss)
