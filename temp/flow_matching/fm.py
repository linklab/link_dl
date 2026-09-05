import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# =========================
# 1) 모델 정의: Velocity(속도장) 예측용 MLP
# =========================
# Flow Matching에서는 "현재 상태 x_tau 와 시간 tau"를 넣으면
# 그 지점에서의 벡터장(= 속도) v_tau 를 예측하는 신경망을 학습한다.
# 여기서는 가장 단순하게 MLP(완전연결)로 구성.
class FlowMatchingNet(nn.Module):
    def __init__(self, data_dim=2, hidden_dim=128):
        super().__init__()

        # 입력 차원: data_dim(예: 2차원 좌표) + tau(스칼라 1개)
        # 출력 차원: data_dim (각 좌표의 변화량/속도 벡터)
        #
        # 예) data_dim=2이면:
        #   입력 = [x, y, tau]  (3차원)
        #   출력 = [vx, vy]     (2차원)
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1, hidden_dim),  # (data_dim+1) -> hidden_dim
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),    # hidden -> hidden
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),    # hidden -> hidden
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim)       # hidden -> data_dim (velocity)
        )

    def forward(self, x, tau):
        """
        x  : (batch, data_dim)  - 현재 위치(상태)
        tau: (batch, 1)         - 시간(0~1 사이)
        반환: (batch, data_dim)  - 예측 velocity
        """
        # 시간 tau를 x에 concat해서 "시간-조건부 벡터장"을 학습하게 만든다.
        # inputs: (batch, data_dim + 1)
        inputs = torch.cat([x, tau], dim=-1)
        return self.net(inputs)


# =========================
# 2) Flow Matching 학습 함수
# =========================
def train_flow_matching(model, real_data, epochs=1000, batch_size=256, lr=1e-3):
    """
    model     : velocity 예측 모델
    real_data : 타겟 데이터 샘플들 (N, data_dim)
    epochs    : 학습 반복 횟수
    batch_size: 미니배치 크기
    lr        : 학습률
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        # -------------------------
        # (1) 타겟 데이터(x1) 배치 샘플링
        # -------------------------
        # real_data에서 랜덤 인덱스를 뽑아 배치를 구성
        idx = torch.randint(low=0, high=len(real_data), size=(batch_size,))
        x1 = real_data[idx]  # (batch, data_dim)  "깨끗한 데이터" (목표 분포 샘플)

        # -------------------------
        # (2) 베이스 분포 노이즈(x0) 샘플링
        # -------------------------
        # x1과 같은 shape로 표준정규분포 N(0, I) 샘플 생성
        x0 = torch.randn_like(x1)  # (batch, data_dim)

        # -------------------------
        # (3) 시간 tau 샘플링
        # -------------------------
        # tau는 0~1 사이에서 균등분포로 뽑는다.
        # (다른 논문/구현에서는 logit-normal 등으로 tau 분포를 바꾸기도 함)
        tau = torch.rand(batch_size, 1)  # (batch, 1)

        # -------------------------
        # (4) 중간 상태 x_tau 생성 (선형 보간)
        # -------------------------
        # x_tau = (1 - tau) * x0 + tau * x1
        #
        # - tau=0이면 x_tau = x0 (순수 노이즈)
        # - tau=1이면 x_tau = x1 (실제 데이터)
        #
        # 즉, 노이즈에서 데이터로 가는 "직선 경로" 위의 점을 랜덤하게 찍는 셈
        x_tau = (1 - tau) * x0 + tau * x1  # (batch, data_dim)

        # -------------------------
        # (5) 정답 velocity(v_gt) 정의
        # -------------------------
        # 위에서 x_tau를 "직선 경로"로 만들었으므로,
        # 그 경로의 속도(derivative)는 상수로 x1 - x0 가 된다.
        #
        # (정확히는 x_tau = x0 + tau*(x1-x0) 이므로 d/dtau x_tau = x1 - x0)
        v_gt = x1 - x0  # (batch, data_dim)

        # -------------------------
        # (6) 모델 예측 velocity(v_pred)
        # -------------------------
        # 모델은 (x_tau, tau)를 받아 그 지점에서의 속도장을 예측하도록 학습된다.
        v_pred = model(x_tau, tau)  # (batch, data_dim)

        # -------------------------
        # (7) Loss 계산 (MSE)
        # -------------------------
        # Flow Matching의 핵심 아이디어(아주 단순화 버전):
        # "경로 위 임의 지점에서 정답 벡터장(속도)과 모델 벡터장을 맞추자"
        #
        # 여기서는 각 차원별 squared error를 평균낸 MSE 사용
        loss = torch.mean((v_pred - v_gt) ** 2)

        # -------------------------
        # (8) 역전파 및 최적화
        # -------------------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

    return losses


# =========================
# 3) 샘플링 함수: ODE를 Euler로 적분해서 x0 -> x1로 이동
# =========================
@torch.no_grad()
def sample(model, n_samples=1000, n_steps=100, data_dim=2):
    """
    학습된 모델을 이용해 노이즈에서 시작해 데이터로 '흘려보내는' 과정.

    - 시작: x ~ N(0, I)
    - 시간: tau = 0 -> 1
    - 업데이트: x_{t+1} = x_t + v(x_t, tau_t) * delta  (Euler step)

    반환:
    - x: 최종 생성 샘플 (n_samples, data_dim)
    - trajectory: 각 스텝의 x를 저장한 리스트 (시각화용)
    """
    model.eval()

    # (1) 초기값: 순수 노이즈
    x = torch.randn(n_samples, data_dim)  # (n_samples, data_dim)

    # (2) 적분 스텝 크기 (0~1 구간을 n_steps로 나눔)
    delta = 1.0 / n_steps

    # 시각화를 위해 매 스텝의 좌표를 저장
    trajectory = [x.clone()]

    # (3) 0 -> 1까지 n_steps번 이동
    for step in range(n_steps):
        # 현재 시간 tau_t (스칼라)을 모든 샘플에 동일하게 부여
        # shape을 (n_samples, 1)로 맞춰 concat 가능하게 함
        tau = torch.ones(n_samples, 1) * (step * delta)  # (n_samples, 1)

        # 현재 위치 x와 시간 tau에서의 velocity 예측
        v = model(x, tau)  # (n_samples, data_dim)

        # Euler 적분: x_{t+1} = x_t + v * delta
        x = x + v * delta

        # trajectory 저장(시각화용)
        trajectory.append(x.clone())

    return x, trajectory


# =========================
# 4) 타겟 데이터 생성 (2D 가우시안 혼합)
# =========================
def create_target_data(n_samples=10000):
    """
    학습 목표가 되는 real_data 생성.
    여기서는 2개의 가우시안 클러스터를 만들어 mixture 형태로 둔다.
    """
    # 클러스터 1: 평균 [2, 2], 표준편차 0.5 정도
    data1 = torch.randn(n_samples // 2, 2) * 0.5 + torch.tensor([2.0, 2.0])

    # 클러스터 2: 평균 [-2, -2], 표준편차 0.5 정도
    data2 = torch.randn(n_samples // 2, 2) * 0.5 + torch.tensor([-2.0, -2.0])

    # 두 클러스터 합치기
    data = torch.cat([data1, data2], dim=0)  # (n_samples, 2)
    return data


# =========================
# 5) 시각화 함수
# =========================
def visualize_results(real_data, generated_data, trajectory, losses):
    """
    real_data      : numpy array (N, 2)
    generated_data : numpy array (M, 2)
    trajectory     : list of torch tensors, 길이 (n_steps+1), 각 원소 (M, 2)
    losses         : 학습 loss 리스트
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # -------------------------
    # (1) 학습 Loss 그래프
    # -------------------------
    axes[0].plot(losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True)

    # -------------------------
    # (2) 실제 데이터 vs 생성 데이터 분포 비교
    # -------------------------
    axes[1].scatter(real_data[:, 0], real_data[:, 1],
                    alpha=0.3, s=10, label='Real Data')
    axes[1].scatter(generated_data[:, 0], generated_data[:, 1],
                    alpha=0.3, s=10, label='Generated Data')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].set_title('Real vs Generated Data')
    axes[1].legend()
    axes[1].grid(True)

    # -------------------------
    # (3) 샘플링 궤적 시각화 (처음 100개 샘플만)
    # -------------------------
    # trajectory는 list이므로 (time, sample, dim) 형태로 stack
    trajectory_array = torch.stack(trajectory).numpy()  # (n_steps+1, n_samples, 2)

    # 각 샘플의 이동 경로를 선으로 그림 (너무 많으면 복잡하니 100개만)
    for i in range(min(100, trajectory_array.shape[1])):
        axes[2].plot(trajectory_array[:, i, 0], trajectory_array[:, i, 1],
                     alpha=0.1, c='blue', linewidth=0.5)

    # 시작점(노이즈) 표시: 빨간 점
    axes[2].scatter(
        trajectory_array[0, :100, 0], trajectory_array[0, :100, 1],
        c='red', s=20, label='Start (Noise)', zorder=5
    )

    # 끝점(데이터로 이동한 결과) 표시: 초록 점
    axes[2].scatter(
        trajectory_array[-1, :100, 0], trajectory_array[-1, :100, 1],
        c='green', s=20, label='End (Data)', zorder=5
    )

    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    axes[2].set_title('Sampling Trajectories')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


# =========================
# 6) 메인 실행
# =========================
if __name__ == "__main__":
    # (1) 데이터 생성
    print("Generating target data...")
    real_data = create_target_data(n_samples=10000)
    print("real_data.shape:", real_data.shape)

    # (2) 모델 생성
    print("Creating model...")
    model = FlowMatchingNet(data_dim=2, hidden_dim=128)

    # (3) 학습
    print("Training Flow Matching model...")
    losses = train_flow_matching(model, real_data, epochs=3000, batch_size=256)

    # (4) 샘플링(생성)
    print("Generating samples...")
    generated_data, trajectory = sample(model, n_samples=1000, n_steps=100)

    # (5) 시각화
    print("Visualizing results...")
    visualize_results(real_data.numpy(), generated_data.numpy(), trajectory, losses)

    print("Done!")