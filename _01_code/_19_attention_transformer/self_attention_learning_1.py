import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from dataclasses import dataclass

from torchinfo import summary


# ============================================
# 설정 클래스
# ============================================
@dataclass
class Config:
    """학습 설정을 담는 데이터 클래스"""
    seq_len: int = 15
    input_dim: int = 32
    embed_dim: int = 64
    num_heads: int = 4
    num_classes: int = 3
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    dropout: float = 0.1

    # Scheduler 관련 (StepLR)
    scheduler_step_size: int = 30
    scheduler_gamma: float = 0.5

    grad_clip_norm: float = 1.0

    # 데이터 생성 관련
    num_train_samples_per_class: int = 400
    num_test_samples_per_class: int = 100
    noise_std: float = 0.5
    signal_strength: float = 0.7
    signal_noise: float = 0.4
    additional_noise: float = 1.0

    # DataLoader 관련
    num_workers: int = 0
    pin_memory: bool = True
    shuffle_train: bool = True  # "설명 출력은 그대로"라서 유지 (데이터 자체는 셔플 안 함)


# ============================================
# PyTorch Dataset 클래스
# ============================================
class SequenceDataset(Dataset):
    """시퀀스 분류를 위한 PyTorch Dataset"""

    def __init__(self, data: torch.Tensor, labels: torch.Tensor):
        assert len(data) == len(labels), "데이터와 레이블 크기가 일치해야 합니다."
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ============================================
# 모델 정의
# ============================================
class SimpleAttentionModel(nn.Module):
    """Self-Attention 기반 시퀀스 분류 모델"""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.input_projection = nn.Linear(config.input_dim, config.embed_dim)

        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(config.embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim * 2, config.embed_dim)
        )
        self.norm2 = nn.LayerNorm(config.embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(config.embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(32, config.num_classes)
        )

    def forward(self, x: torch.Tensor):
        x = self.input_projection(x)

        attn_output, attn_weights = self.self_attention(x, x, x)
        x = self.norm1(x + attn_output)

        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        x = x.mean(dim=1)  # Global Average Pooling
        logits = self.classifier(x)

        return logits, attn_weights


# ============================================
# 데이터 생성 모듈 (간단화)
# ============================================
class DataGenerator:
    """시퀀스 분류 데이터 생성기"""

    def __init__(self, config: Config):
        self.c = config

    def generate_dataset(self, num_samples_per_class: int) -> SequenceDataset:
        data, labels = self._generate_raw_data(num_samples_per_class)
        return SequenceDataset(data, labels)

    def _generate_raw_data(self, num_samples_per_class: int):
        data = []
        labels = []

        for class_idx in range(self.c.num_classes):
            for _ in range(num_samples_per_class):
                data.append(self._generate_sample(class_idx))
                labels.append(class_idx)

        data = torch.stack(data)                   # (N, seq_len, input_dim)
        labels = torch.tensor(labels, dtype=torch.long)  # (N,)

        return data, labels

    def _generate_sample(self, class_idx: int) -> torch.Tensor:
        c = self.c
        seq = torch.randn(c.seq_len, c.input_dim) * c.noise_std

        if class_idx == 0:
            seq[:, 0] = torch.randn(c.seq_len) * c.signal_noise + c.signal_strength
            seq[:, 1] = torch.randn(c.seq_len) * c.noise_std + 0.1
            seq[:, 2] = torch.randn(c.seq_len) * c.noise_std - 0.2
        elif class_idx == 1:
            seq[:, 0] = torch.randn(c.seq_len) * c.noise_std - 0.2
            seq[:, 1] = torch.randn(c.seq_len) * c.signal_noise + c.signal_strength
            seq[:, 2] = torch.randn(c.seq_len) * c.noise_std + 0.1
        else:
            seq[:, 0] = torch.randn(c.seq_len) * c.noise_std + 0.1
            seq[:, 1] = torch.randn(c.seq_len) * c.noise_std - 0.2
            seq[:, 2] = torch.randn(c.seq_len) * c.signal_noise + c.signal_strength

        seq = seq + torch.randn(c.seq_len, c.input_dim) * c.additional_noise
        return seq


# ============================================
# 데이터 분석 모듈 (출력 유지)
# ============================================
class DataAnalyzer:
    """데이터 통계 및 분석 도구"""

    @staticmethod
    def print_dataset_info(train_dataset: SequenceDataset, test_dataset: SequenceDataset):
        print("\n[데이터셋 정보]")
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")

        sample_data, sample_label = train_dataset[0]
        print(f"Sample data shape: {sample_data.shape}")
        print(f"Sample label type: {type(sample_label.item())}")

    @staticmethod
    def print_class_distribution(dataset: SequenceDataset, dataset_name: str = "Dataset"):
        labels = dataset.labels
        num_classes = labels.max().item() + 1

        print(f"\n[{dataset_name} 클래스 분포]")
        for i in range(num_classes):
            count = (labels == i).sum().item()
            percentage = count / len(labels) * 100
            print(f"  Class {i}: {count} samples ({percentage:.1f}%)")

    @staticmethod
    def print_feature_statistics(dataset: SequenceDataset, num_features: int = 3):
        data = dataset.data
        labels = dataset.labels
        num_classes = labels.max().item() + 1

        print(f"\n[각 클래스의 주요 특징(dim 0~{num_features - 1})의 평균]")
        for class_idx in range(num_classes):
            class_data = data[labels == class_idx]
            print(f"  Class {class_idx}:")
            for dim_idx in range(num_features):
                dim_mean = class_data[:, :, dim_idx].mean().item()
                print(f"    Dim {dim_idx}: {dim_mean:>6.3f}")


# ============================================
# 학습 모듈 (train_batch 분리 제거해서 단순화)
# ============================================
class Trainer:
    """모델 학습 담당 클래스"""

    def __init__(self, model: nn.Module, config: Config, device: torch.device):
        self.model = model
        self.c = config
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma
        )

        print(f"\n[Scheduler 정보]")
        print(f"  Type: StepLR")
        print(f"  Step size: {config.scheduler_step_size} epochs")
        print(f"  Gamma: {config.scheduler_gamma}")
        print(f"  초기 LR: {config.learning_rate}")

        self.train_losses, self.train_accuracies = [], []
        self.test_losses, self.test_accuracies = [], []
        self.learning_rates = []
        self.best_test_acc = 0.0

    def train_epoch(self, train_loader: DataLoader):
        self.model.train()

        epoch_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            logits, _ = self.model(x)
            loss = self.criterion(logits, y)

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.c.grad_clip_norm)
            self.optimizer.step()

            epoch_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        avg_loss = epoch_loss / len(train_loader)
        acc = correct / total * 100
        return avg_loss, acc

    def evaluate(self, test_loader: DataLoader):
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits, _ = self.model(x)
                loss = self.criterion(logits, y)

                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        avg_loss = total_loss / len(test_loader)
        acc = correct / total * 100
        return avg_loss, acc

    def train(self, train_loader: DataLoader, test_loader: DataLoader):
        print(f"\n{'=' * 60}")
        print("학습 시작")
        print(f"{'=' * 60}\n")

        for epoch in range(self.c.num_epochs):
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)

            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            test_loss, test_acc = self.evaluate(test_loader)
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_acc)

            if test_acc > self.best_test_acc:
                self.best_test_acc = test_acc

            self.scheduler.step()

            print(f"Epoch [{epoch + 1}/{self.c.num_epochs}]")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
            print(f"  Best Test Acc: {self.best_test_acc:.2f}% | LR: {current_lr:.6f}")
            print()


# ============================================
# 시각화 모듈 (출력 유지, 코드 단순화)
# ============================================
class Visualizer:
    """학습 결과 시각화 도구"""

    @staticmethod
    def plot_training_results(trainer: Trainer, test_dataset: SequenceDataset, device: torch.device,
                              save_path: str = 'attention_training_results.png'):
        fig = plt.figure(figsize=(24, 5))

        # Loss
        plt.subplot(1, 4, 1)
        plt.plot(trainer.train_losses, label='Train Loss', linewidth=2, alpha=0.8)
        plt.plot(trainer.test_losses, label='Test Loss', linewidth=2, alpha=0.8)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Test Loss', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # Accuracy
        plt.subplot(1, 4, 2)
        plt.plot(trainer.train_accuracies, label='Train Accuracy', linewidth=2, alpha=0.8)
        plt.plot(trainer.test_accuracies, label='Test Accuracy', linewidth=2, alpha=0.8)
        plt.axhline(y=trainer.best_test_acc, linestyle='--', linewidth=1.5,
                    label=f'Best Test: {trainer.best_test_acc:.2f}%')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Training and Test Accuracy', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 105])

        # LR
        plt.subplot(1, 4, 3)
        plt.plot(trainer.learning_rates, label='Learning Rate', linewidth=2, alpha=0.8)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        # Attention heatmap
        plt.subplot(1, 4, 4)
        with torch.no_grad():
            x, _ = test_dataset[0]
            x = x.unsqueeze(0).to(device)
            _, attn = trainer.model(x)
            attn = attn[0].cpu().numpy()

        im = plt.imshow(attn, cmap='viridis', aspect='auto')
        plt.colorbar(im, label='Attention Weight')
        plt.xlabel('Key Position', fontsize=12)
        plt.ylabel('Query Position', fontsize=12)
        plt.title('Attention Weights Heatmap', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 그래프가 '{save_path}'로 저장되었습니다.")


# ============================================
# 평가 모듈 (출력 유지, 내부 구조 약간 단순화)
# ============================================
class Evaluator:
    """모델 성능 평가 도구"""

    @staticmethod
    def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device, num_classes: int = 3):
        print(f"\n{'=' * 60}")
        print("클래스별 성능 분석")
        print(f"{'=' * 60}")

        model.eval()
        preds, trues = [], []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                logits, _ = model(x)
                preds.append(logits.argmax(dim=1).cpu())
                trues.append(y)

        preds = torch.cat(preds)
        trues = torch.cat(trues)

        cm = torch.zeros(num_classes, num_classes, dtype=torch.int)
        for t, p in zip(trues, preds):
            cm[t, p] += 1

        # Confusion Matrix 출력
        print("\nConfusion Matrix:")
        print("         Predicted")
        print("        ", end="")
        for i in range(num_classes):
            print(f" {i:3d} ", end="")
        print("\n       +" + "-" * (num_classes * 5 + 1))
        for i in range(num_classes):
            print(f"True {i} |", end="")
            for j in range(num_classes):
                print(f" {cm[i][j]:3d} ", end="")
            print()

        # 클래스별 메트릭 출력
        print("\n")
        for k in range(num_classes):
            tp = cm[k, k].item()
            fp = cm[:, k].sum().item() - tp
            fn = cm[k, :].sum().item() - tp
            total = cm[k, :].sum().item()
            correct = tp

            acc = correct / total * 100 if total > 0 else 0.0
            precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            print(f"Class {k}:")
            print(f"  Total: {total} | Correct: {correct} | Accuracy: {acc:.2f}%")
            print(f"  Precision: {precision:.2f}% | Recall: {recall:.2f}% | F1: {f1:.2f}%")


# ============================================
# 유틸리티 함수 (필수 출력 유지)
# ============================================
def create_dataloaders(train_dataset: SequenceDataset, test_dataset: SequenceDataset, config: Config):
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle_train,   # loader shuffle은 그대로 (출력/설정 유지 목적)
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    print(f"\n[DataLoader 정보]")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Batch size: {config.batch_size}")

    return train_loader, test_loader

# ============================================
# 전체 데이터 샘플 출력 유틸리티
# ============================================
def print_all_dataset_samples(
    dataset,
    dataset_name: str,
    num_head_features: int = 4,
    num_tail_features: int = 4
):
    """
    모든 데이터 샘플의 내부 내용을 출력
    - 각 time-step마다 feature 앞/뒤 일부만 출력
    - 실제 feature 값 그대로 출력
    """
    print(f"\n{'=' * 60}")
    print(f"{dataset_name} DATASET - ALL SAMPLES (PARTIAL FEATURES)")
    print(f"{'=' * 60}")

    for idx in range(len(dataset)):
        x, y = dataset[idx]  # x: (seq_len, input_dim)

        print(f"\n[{dataset_name} Sample {idx}]")
        print(f"  Label: {int(y)}")
        print(f"  Sequence Length: {x.shape[0]}")
        print(f"  Input Dim: {x.shape[1]}")
        print(f"  Time-step wise feature values:")

        for t in range(x.shape[0]):
            step = x[t]  # (input_dim,)

            head = step[:num_head_features]
            tail = step[-num_tail_features:]

            head_str = ", ".join(f"{v:+6.3f}" for v in head)
            tail_str = ", ".join(f"{v:+6.3f}" for v in tail)

            print(
                f"    t={t:02d} | "
                f"[ {head_str}, ... , {tail_str} ]"
            )

# ============================================
# 메인 실행 함수
# ============================================
def main():
    print("=" * 60)
    print("Self-Attention 학습 예제 (StepLR Scheduler)")
    print("=" * 60)

    config = Config()

    print("\n💡 학습 목표:")
    print("   - 적절한 난이도의 시퀀스 패턴 인식")
    print("   - PyTorch Dataset과 DataLoader 활용")
    print("   - StepLR을 사용한 간단한 학습률 스케줄링\n")
    print("-" * 60)

    # 데이터 생성
    print("\n[데이터 생성]")
    gen = DataGenerator(config)
    train_dataset = gen.generate_dataset(config.num_train_samples_per_class)
    test_dataset = gen.generate_dataset(config.num_test_samples_per_class)

    print_all_dataset_samples(train_dataset, "train")
    print_all_dataset_samples(test_dataset, "test")

    # 데이터 분석
    analyzer = DataAnalyzer()
    analyzer.print_dataset_info(train_dataset, test_dataset)
    analyzer.print_class_distribution(train_dataset, "Train")
    analyzer.print_class_distribution(test_dataset, "Test")
    analyzer.print_feature_statistics(train_dataset)

    # DataLoader
    train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, config)

    # 디바이스
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Device]: {device}")

    # 모델
    model = SimpleAttentionModel(config).to(device)
    summary(model)

    # 학습
    trainer = Trainer(model, config, device)
    trainer.train(train_loader, test_loader)

    # 최종 결과 출력
    print(f"\n{'=' * 60}")
    print("학습 완료!")
    print(f"{'=' * 60}")
    print(f"최종 Train Accuracy: {trainer.train_accuracies[-1]:.2f}%")
    print(f"최종 Test Accuracy: {trainer.test_accuracies[-1]:.2f}%")
    print(f"최고 Test Accuracy: {trainer.best_test_acc:.2f}%")
    print(f"최종 Learning Rate: {trainer.learning_rates[-1]:.6f}")

    # 시각화
    Visualizer.plot_training_results(trainer, test_dataset, device)

    # 평가
    Evaluator.evaluate_model(model, test_loader, device, config.num_classes)

    print(f"\n{'=' * 60}")
    print("✅ 학습 및 분석 완료!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()