# https://www.python-engineer.com/courses/pytorchbeginner/03-autograd/
import torch

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

  print("weights:", weights)

  # this is important! It affects the final weights & output
  weights.grad.zero_()

model_output = (weights * 3).sum()
print("\n[model_output final]:", model_output)

print("#" * 50, 1)

weights = torch.ones(4, requires_grad=True)
optimizer = torch.optim.SGD([weights], lr=0.1)

for epoch in range(3):
  # just a dummy example
  model_output = (weights * 3).sum()
  print("[model_output {0}]:".format(epoch), model_output)
  model_output.backward()

  print("weights.grad:", weights.grad)

  # optimize model, i.e. adjust weights & empty gradients
  optimizer.step()
  optimizer.zero_grad()

  print("weights:", weights)

model_output = (weights * 3).sum()
print("\n[model_output final]:", model_output)
