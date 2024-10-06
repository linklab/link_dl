import matplotlib.pyplot as plt
import torch
import torch.nn as nn

input_t = torch.arange(start=-8, end=8.1, step=0.1)
# print(input_t)
# >>> tensor([-8.0000e+00, -7.9000e+00, -7.8000e+00, ...,  7.8000e+00,  7.9000e+00, 8.0000e+00])

activation_list = [
  nn.Tanh(),
  nn.Sigmoid(),

  nn.Softplus(),
  nn.ReLU(),

  nn.ReLU6(),
  nn.ELU(alpha=1.0),

  nn.LeakyReLU(negative_slope=0.01),
  nn.RReLU(),
]

fig, axs = plt.subplots(4, 2)
x = 1280 / fig.dpi  # 가로 길이 (1280 pixel)
y = 2080 / fig.dpi  # 세로 길이 (9960 pixel)
fig.set_figwidth(x)
fig.set_figheight(y)

for idx, activation_func in enumerate(activation_list):
  i, j = divmod(idx, 2)
  axs[i][j].set_title(type(activation_func).__name__)

  output_t = activation_func(input_t)

  axs[i][j].grid()
  axs[i][j].plot(input_t.numpy(), input_t.numpy(), 'k', linewidth=1)
  axs[i][j].plot([-8, 8], [0, 0], 'k', linewidth=1)
  axs[i][j].plot([0, 0], [-8, 8], 'k', linewidth=1)
  axs[i][j].plot(input_t.numpy(), output_t.numpy(), 'r', linewidth=3)

plt.show()
