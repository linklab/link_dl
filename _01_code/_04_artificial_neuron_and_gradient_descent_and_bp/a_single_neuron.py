import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device("cpu")
# device = torch.device("cuda:0") # GPU에서 실행하려면 이 주석을 제거


class SimpleDataset(Dataset):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    X = [[0.5, 0.9], [14.0, 12.0], [15.0, 13.6],
         [28.0, 22.8], [11.0, 8.1], [8.0, 7.1],
         [3.0, 2.9], [4.0, 0.1], [6.0, 5.3],
         [13.0, 12.0], [21.0, 19.9], [-1.0, 1.5]]

    y = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4, 29.1]

    self.X = torch.tensor(X, dtype=torch.float, device=device)
    self.y = torch.tensor(y, dtype=torch.float, device=device)
    self.y = self.y * 0.01

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]

  def __str__(self):
    str = "Data Size: {0}, Input Shape: {1}, Target Shape: {2}".format(
      len(self.X), self.X.shape, self.y.shape
    )
    return str


def model(X, W, b):
  # print(X.shape)  # >>> torch.Size([12, 2])
  # print(W.shape)  # >>> torch.Size([2])
  # print(b.shape)  # >>> torch.Size([1])
  u = torch.sum(X * W, dim=1) + b
  z = activate(u)
  return z


def activate(u):
  return F.sigmoid(u)


def loss_fn(y_pred, y):
  loss = torch.square(y_pred - y).mean()    # print(loss.shape) >>> torch.Size([])
  assert loss.shape == () or loss.shape == (1,)
  return loss


def gradient(W, b, X, y_pred, y):
  # W.shape: (2,), b.shape: (1,)
  # X.shape: (12, 2), y.shape: (12,)
  dl_dy = 2 * (y_pred - y)
  dl_dy = dl_dy.unsqueeze(dim=-1)  # dl_dy_pred.shape: [12, 1]

  dy_ds = 1.0

  z = torch.sum(X * W, dim=-1) + b
  ds_dz = activate(z) * (1.0 - activate(z))
  ds_dz = ds_dz.unsqueeze(dim=-1)  # ds_dz_pred.shape: [12, 1]

  W_grad = torch.mean(dl_dy * dy_ds * ds_dz * X, dim=0)
  b_grad = torch.mean(dl_dy * dy_ds * ds_dz * 1.0, dim=0)

  return W_grad, b_grad


def learn(W, b, train_data_loader):
  MAX_EPOCHS = 20_000
  LEARNING_RATE = 0.01

  for epoch in range(0, MAX_EPOCHS):
    batch = next(iter(train_data_loader))
    input, target = batch
    y_pred = model(input, W, b)     # y_pred.shape: (12,)
    loss = loss_fn(y_pred, target)

    W_grad, b_grad = gradient(W, b, input, y_pred, target)

    if epoch % 100 == 0:
      print("[Epoch:{0:6,}] loss:{1:8.5f}, w0:{2:6.3f}, w1:{3:6.3f}, b:{4:6.3f}".format(
        epoch, loss.item(), W[0].item(), W[1].item(), b.item()
      ), end=", ")
      print("W.grad: {0}, b.grad:{1}".format(W_grad, b_grad))

    W = W - LEARNING_RATE * W_grad
    b = b - LEARNING_RATE * b_grad


def main():
  W = torch.ones((2,))
  b = torch.zeros((1,))

  simple_dataset = SimpleDataset()
  train_data_loader = DataLoader(dataset=simple_dataset, batch_size=len(simple_dataset))
  batch = next(iter(train_data_loader))

  input, target = batch

  y_pred = model(input, W, b)
  print(y_pred.shape)
  print(y_pred)

  loss = loss_fn(y_pred, target)
  print(loss)

  learn(W, b, train_data_loader)


if __name__ == "__main__":
  main()
