import torch
from _04_learning_and_autograd.a_single_neuron import model, loss_fn, get_data


def learn(W, b, X, y):
    MAX_EPOCHS = 20_000
    LEARNING_RATE = 0.01

    from torch import optim
    optimizer = optim.SGD([W, b], lr=LEARNING_RATE)

    for epoch in range(0, MAX_EPOCHS):
        y_pred = model(X, W, b)
        loss = loss_fn(y_pred, y)

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

    X, y = get_data()

    learn(W, b, X, y)


if __name__ == "__main__":
    main()
