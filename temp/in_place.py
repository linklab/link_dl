import torch
from torch.nn import Linear
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

model = Linear(10, 20)

# 옵티마이저 및 스케줄러 정의
optimizer = SGD(model.parameters(), lr=0.1)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

for epoch in range(16):
    scheduler.step()  # 에포크 종료 후 학습률 갱신
    print(f"Epoch {epoch}, Learning Rate: {optimizer.param_groups[0]['lr']}")