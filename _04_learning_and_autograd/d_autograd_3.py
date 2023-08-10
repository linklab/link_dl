# https://www.python-engineer.com/courses/pytorchbeginner/03-autograd/
import torch

a = torch.randn(2, 2, requires_grad=True)
print(a.requires_grad)
with torch.no_grad():
    print(a.requires_grad)
    b = a * 2
    print(b.requires_grad)

print("#" * 50, 1)

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
