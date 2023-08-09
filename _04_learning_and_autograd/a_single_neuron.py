import torch
import torch.nn.functional as F

device = torch.device("cpu")
# device = torch.device("cuda:0") # GPU에서 실행하려면 이 주석을 제거

def get_data():
    X = [[0.5, 0.9], [14.0, 12.0], [15.0, 13.6],
         [28.0, 22.8], [11.0, 8.1], [8.0, 7.1],
         [3.0, 2.9], [4.0, 0.1], [6.0, 5.3],
         [13.0, 12.0], [21.0, 19.9], [1.0, 1.5]]

    y = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4, 29.1]

    X = torch.tensor(X, dtype=torch.float, device=device)
    y = torch.tensor(y, dtype=torch.float, device=device)

    y_normalized = y * 0.01

    return X, y_normalized


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
    loss = torch.square(y_pred - y).mean()
    assert loss.shape == () or loss.shape == (1,)
    return loss


def gradient(W, b, X, y):
    y_pred = model(X, W, b)
    dl_dy_pred = 2 * (y_pred - y)
    dl_dy_pred = dl_dy_pred.unsqueeze(dim=-1)   # dl_dy_pred.shape: [12, 1]

    dy_pred_df = 1.0

    u = torch.sum(X * W, dim=1) + b
    df_du = activate(u) * (1.0 - activate(u))
    df_du = df_du.unsqueeze(dim=-1)             # dl_dy_pred.shape: [12, 1]

    W_grad = torch.mean(dl_dy_pred * dy_pred_df * df_du * X, dim=0)
    b_grad = torch.mean(dl_dy_pred * dy_pred_df * df_du * 1.0, dim=0)

    return W_grad, b_grad


def learn(W, b, X, y):
    MAX_EPOCHS = 20_000
    LEARNING_RATE = 0.01

    for epoch in range(0, MAX_EPOCHS):
        y_pred = model(X, W, b)
        loss = loss_fn(y_pred, y)

        W_grad, b_grad = gradient(W, b, X, y)

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

    X, y = get_data()
    y_pred = model(X, W, b)
    print(y_pred.shape)
    print(y_pred)

    loss = loss_fn(y_pred, y)
    print(loss)

    learn(W, b, X, y)


if __name__ == "__main__":
    main()
