from torch.optim.lr_scheduler import StepLR
from torch import nn, optim

model = nn.Linear(10, 10)
optimizer = optim.Adam(model.parameters(), lr=0.1)

# 옵티마이저 및 스케줄러 정의
scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

for epoch in range(1, 10):
    #train(...)         # 학습 수행
    scheduler.step()  # 에포크 종료 후 학습률 갱신
    print(f"Epoch {epoch}, Learning Rate: {optimizer.param_groups[0]['lr']}")

