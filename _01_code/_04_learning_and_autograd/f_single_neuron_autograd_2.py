import torch
from torch.utils.data import DataLoader

from a_single_neuron import model, loss_fn, SimpleDataset


def learn(W, b, train_data_loader):
  MAX_EPOCHS = 20_000
  LEARNING_RATE = 0.01

  from torch import optim
  optimizer = optim.SGD([W, b], lr=LEARNING_RATE)

  for epoch in range(0, MAX_EPOCHS):
    batch = next(iter(train_data_loader))
    y_pred = model(batch["input"], W, b)
    loss = loss_fn(y_pred, batch["target"])

    loss.backward()

    if epoch % 100 == 0:
      print("[Epoch:{0:6,}] loss:{1:8.5f}, w0:{2:6.3f}, w1:{3:6.3f}, b:{4:6.3f}".format(
        epoch, loss.item(), W[0].item(), W[1].item(), b.item()
      ), end=", ")
      print("W.grad: {0}, b.grad:{1}".format(W.grad, b.grad))

    optimizer.step()
    optimizer.zero_grad()


def main():
  W = torch.ones((2,), requires_grad=True)
  b = torch.zeros((1,), requires_grad=True)

  simple_dataset = SimpleDataset()
  train_data_loader = DataLoader(dataset=simple_dataset, batch_size=len(simple_dataset))
  learn(W, b, train_data_loader)


if __name__ == "__main__":
  main()
