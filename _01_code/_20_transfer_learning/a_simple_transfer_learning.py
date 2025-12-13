from torchvision import models
from torch import nn, optim

# ResNet18 모델 로드
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

for name, param in model.named_parameters():
    print(name, param.shape)
    param.requires_grad = False     # 기존 가중치를 고정

print(model.fc)

class_names = ['AAA', 'BBB', 'CCC']

# 새로운 fc 출력층 정의 (기존 fc 의 in_features 활용)
model.fc = nn.Linear(model.fc.in_features, len(class_names))

# Feature Extraction
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Fine-Tuning (모델 전체 학습 시)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
