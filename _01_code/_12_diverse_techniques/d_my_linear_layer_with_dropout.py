import torch
from torch import nn


class MyLinearWithActivationAndDropout(nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    self.weight = nn.Parameter(torch.randn(out_features, in_features))
    self.bias = nn.Parameter(torch.randn(out_features))
    self.activation = nn.Sigmoid()
    self.drop_prob = 0.3

  def forward(self, input, is_train):
    z = input @ self.weight.t() + self.bias
    a = self.activation(z)

    if is_train:  # dropout only if model is trained
      a_masked, a_masked_and_scaled = self.dropout(a)
      return a_masked, a_masked_and_scaled
    else:
      return a

  def dropout(self, a):
    # Step 1: initialize matrix r, where its shape is same as the above a tensor
    r = torch.rand_like(a)

    # Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)
    mask = (r < self.drop_prob).to(torch.int)
    print(mask, "!!!")

    # Step 3: shut down some neurons of A1
    a_masked = mask * a

    # Step 4: scale the value of neurons that haven't been shut down
    a_masked_and_scaled = a_masked / (1 - self.drop_prob)  # It is called 'Inverted Dropout'

    return a_masked, a_masked_and_scaled


if __name__ == "__main__":
  my_linear = MyLinearWithActivationAndDropout(in_features=4, out_features=7)

  batch_input = torch.randn(3, 4)
  batch_output, batch_input_scaled = my_linear(batch_input, is_train=True)

  print("input.shape:", batch_input.shape)
  print("output.shape:", batch_output.shape)
  print(batch_output, torch.sum(batch_output, dim=0))
  print(batch_input_scaled, torch.sum(batch_input_scaled, dim=0))