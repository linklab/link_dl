import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# 간단한 MLP 네트워크 (velocity 예측)
class FlowMatchingNet(nn.Module):
    def __init__(self, data_dim=2, hidden_dim=128):
        super().__init__()
        # 입력: [x_tau, tau] -> 출력: velocity
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim)
        )

    def forward(self, x, tau):
        # x: (batch, data_dim), tau: (batch, 1)
        inputs = torch.cat([x, tau], dim=-1)
        return self.net(inputs)


# Flow Matching 학습 함수
def train_flow_matching(model, real_data, epochs=1000, batch_size=256, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        # 배치 샘플링
        idx = torch.randint(0, len(real_data), size=(batch_size,))
        x1 = real_data[idx]  # 깨끗한 데이터

        # 노이즈 샘플링
        x0 = torch.randn_like(x1)  # N(0, I)

        # 시간 샘플링 (uniform or logit-normal)
        tau = torch.rand(batch_size, 1)

        # 섞인 데이터 생성 (식 1)
        x_tau = (1 - tau) * x0 + tau * x1

        # Ground truth velocity
        v_gt = x1 - x0

        # 예측
        v_pred = model(x_tau, tau)

        # Loss 계산 (식 1)
        loss = torch.mean((v_pred - v_gt) ** 2)

        # 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

    return losses


# 샘플링 함수 (식 2)
@torch.no_grad()
def sample(model, n_samples=1000, n_steps=100, data_dim=2):
    model.eval()

    # 순수 노이즈에서 시작
    x = torch.randn(n_samples, data_dim)
    delta = 1.0 / n_steps  # 스텝 크기

    trajectory = [x.clone()]

    # K번의 스텝으로 iterative 변환
    for step in range(n_steps):
        tau = torch.ones(n_samples, 1) * (step * delta)

        # velocity 예측
        v = model(x, tau)

        # 다음 위치로 이동 (식 2)
        x = x + v * delta
        trajectory.append(x.clone())

    return x, trajectory


# 타겟 데이터 생성 (예: 두 개의 가우시안 혼합)
def create_target_data(n_samples=10000):
    # 두 개의 클러스터
    data1 = torch.randn(n_samples // 2, 2) * 0.5 + torch.tensor([2.0, 2.0])
    data2 = torch.randn(n_samples // 2, 2) * 0.5 + torch.tensor([-2.0, -2.0])
    data = torch.cat([data1, data2], dim=0)
    return data


# 시각화 함수
def visualize_results(real_data, generated_data, trajectory, losses):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. 학습 Loss
    axes[0].plot(losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True)

    # 2. 실제 vs 생성 데이터
    axes[1].scatter(real_data[:, 0], real_data[:, 1], alpha=0.3, s=10, label='Real Data')
    axes[1].scatter(generated_data[:, 0], generated_data[:, 1], alpha=0.3, s=10, label='Generated Data')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].set_title('Real vs Generated Data')
    axes[1].legend()
    axes[1].grid(True)

    # 3. 샘플링 궤적 (처음 100개만)
    trajectory_array = torch.stack(trajectory).numpy()
    for i in range(min(100, trajectory_array.shape[1])):
        axes[2].plot(trajectory_array[:, i, 0], trajectory_array[:, i, 1], alpha=0.1, c='blue', linewidth=0.5)
    axes[2].scatter(
        trajectory_array[0, :100, 0], trajectory_array[0, :100, 1], c='red', s=20, label='Start (Noise)', zorder=5
    )
    axes[2].scatter(
        trajectory_array[-1, :100, 0], trajectory_array[-1, :100, 1], c='green', s=20, label='End (Data)', zorder=5
    )
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    axes[2].set_title('Sampling Trajectories')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


# 메인 실행
if __name__ == "__main__":
    # 데이터 생성
    print("Generating target data...")
    real_data = create_target_data(n_samples=10000)
    print("real_data.shape:", real_data.shape)

    # 모델 생성
    print("Creating model...")
    model = FlowMatchingNet(data_dim=2, hidden_dim=128)

    # 학습
    print("Training Flow Matching model...")
    losses = train_flow_matching(model, real_data, epochs=1000, batch_size=256)

    # 샘플링
    print("Generating samples...")
    generated_data, trajectory = sample(model, n_samples=1000, n_steps=100)

    # 시각화
    print("Visualizing results...")
    visualize_results(real_data.numpy(), generated_data.numpy(), trajectory, losses)

    print("Done!")