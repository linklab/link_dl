import torch
from torch.utils.data import DataLoader

from _01_code._04_artificial_neuron_and_gradient_descent_and_bp.a_single_neuron import model, loss_fn, SimpleDataset


def learn(W, b, train_data_loader):
  MAX_EPOCHS = 30_000
  LEARNING_RATE = 0.01

  print("[Epoch:{0:6,}] loss: {1:9}, w0:{2:12.9f}, w1:{3:12.9f}, b:{4:12.9f}".format(
    0, 'N/A', W[0].item(), W[1].item(), b.item()
  ))
  for epoch in range(1, MAX_EPOCHS + 1):
    batch = next(iter(train_data_loader))
    inputs, targets = batch
    # inputs.shape: torch.Size([12, 2])
    # target.shape: torch.Size([12])

    y_pred = model(inputs, W, b)
    loss = loss_fn(y_pred, targets)

    loss.backward()

    with torch.no_grad():
      W -= LEARNING_RATE * W.grad
      b -= LEARNING_RATE * b.grad
      W.grad.zero_()
      b.grad.zero_()

    if epoch % 10 == 0:
      print("[Epoch:{0:6,}] loss:{1:10.7f}, w0:{2:12.9f}, w1:{3:12.9f}, b:{4:12.9f}".format(
        epoch, loss.item(), W[0].item(), W[1].item(), b.item()
      ))

def main():
  W = torch.ones((2,), requires_grad=True)
  b = torch.zeros((1,), requires_grad=True)

  simple_dataset = SimpleDataset()
  train_data_loader = DataLoader(dataset=simple_dataset, batch_size=len(simple_dataset))
  batch = next(iter(train_data_loader))

  input, target = batch
  print(input.shape, target.shape, "!!!!!")

  y_pred = model(input, W, b)
  print(y_pred.shape)
  print("y_pred:", y_pred)
  print("target:", target)

  loss = loss_fn(y_pred, target)
  print("loss:", loss)

  print("#" * 50)

  learn(W, b, train_data_loader)

  print("#" * 50)

  y_pred = model(input, W, b)
  print("y_pred:", y_pred)
  print("target:", target)
  loss = loss_fn(y_pred, target)
  print("loss:", loss)


if __name__ == "__main__":
  main()
