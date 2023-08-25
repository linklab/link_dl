import torch
import math

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # GPU에서 실행하려면 이 주석을 제거하세요

# 입력값과 출력값을 갖는 텐서들을 생성합니다.
# requires_grad=False가 기본값으로 설정되어 역전파 단계 중에 이 텐서들에 대한 변화도를
# 계산할 필요가 없음을 나타냅니다.
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# 가중치를 갖는 임의의 텐서를 생성합니다. 3차 다항식이므로 4개의 가중치가 필요합니다:
# y = a + b x + c x^2 + d x^3
# requires_grad=True로 설정하여 역전파 단계 중에 이 텐서들에 대한 변화도를 계산할 필요가
# 있음을 나타냅니다.
a = torch.randn(size=(), device=device, dtype=dtype, requires_grad=True)
b = torch.randn(size=(), device=device, dtype=dtype, requires_grad=True)
c = torch.randn(size=(), device=device, dtype=dtype, requires_grad=True)
d = torch.randn(size=(), device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(2000):
  # 순전파 단계: 텐서들 간의 연산을 사용하여 예측값 y를 계산합니다.
  y_pred = a + b * x + c * x ** 2 + d * x ** 3

  # 텐서들간의 연산을 사용하여 손실(loss)을 계산하고 출력합니다.
  # 이 때 손실은 (1,) shape을 갖는 텐서입니다.
  # loss.item() 으로 손실이 갖고 있는 스칼라 값을 가져올 수 있습니다.
  loss = (y_pred - y).pow(2).sum()
  if t % 100 == 99:
    print(t, loss.item())

  # autograd 를 사용하여 역전파 단계를 계산합니다. 이는 requires_grad=True를 갖는
  # 모든 텐서들에 대한 손실의 변화도를 계산합니다.
  # 이후 a.grad와 b.grad, c.grad, d.grad는 각각 a, b, c, d에 대한 손실의 변화도를
  # 갖는 텐서가 됩니다.
  loss.backward()

  # 경사하강법(gradient descent)을 사용하여 가중치를 직접 갱신합니다.
  # torch.no_grad()로 감싸는 이유는, 가중치들이 requires_grad=True 지만
  # autograd에서는 이를 추적하지 말아야 되기 때문입니다.
  with torch.no_grad():
    a -= learning_rate * a.grad
    b -= learning_rate * b.grad
    c -= learning_rate * c.grad
    d -= learning_rate * d.grad

    # 가중치 갱신 후에는 변화도를 직접 0으로 만듭니다.
    a.grad = None
    b.grad = None
    c.grad = None
    d.grad = None

print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
