import torch

w = torch.tensor([2.0], requires_grad=True)

# 첫 번째 loss
loss1 = (w - 3)**2
loss1.backward()
print("After first backward:", w.grad)  # >>> tensor([-2.])

# 두 번째 loss (같은 w에 대해 다시 backward)
loss2 = (w - 5)**2
loss2.backward()
print("After second backward:", w.grad)  # >>> tensor([-6.])  (-2 + -4 누적됨)