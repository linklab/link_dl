import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
from torchinfo import summary

# self_attention_learning_1.py에서 재사용
import self_attention_learning_1 as base


# ============================================
# 설정 클래스 (1.py Config를 상속 + 가변 길이 옵션 추가)
# ============================================
@dataclass
class Config(base.Config):
    # 샘플별 시퀀스 길이 범위 (새로 추가)
    min_seq_len: int = 5
    max_seq_len: int = 25


# ============================================
# 가변 길이 Dataset (1.py의 SequenceDataset을 "형식만" 재사용하지 않고, 새로 정의)
# - 이유: 1.py의 SequenceDataset은 data가 텐서 (N, L, D) 고정형을 가정
# - 여기서는 list[Tensor(L_i, D)] 형태가 필요
# ============================================
class VariableLengthSequenceDataset(torch.utils.data.Dataset):
    """가변 길이 시퀀스 분류용 Dataset"""

    def __init__(self, sequences, labels):
        assert len(sequences) == len(labels), "데이터와 레이블 크기가 일치해야 합니다."
        self.sequences = sequences  # list of (seq_len_i, input_dim)
        self.labels = labels        # (N,)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# ============================================
# 가변 길이 데이터 생성기 (1.py DataGenerator 스타일을 재사용해서 작성)
# ============================================
class DataGeneratorVarLen:
    """샘플마다 seq_len이 다른 데이터 생성기"""

    def __init__(self, config: Config):
        self.c = config

    def generate_dataset(self, num_samples_per_class: int) -> VariableLengthSequenceDataset:
        seqs, labels = self._generate_raw_data(num_samples_per_class)
        return VariableLengthSequenceDataset(seqs, labels)

    def _generate_raw_data(self, num_samples_per_class: int):
        seqs = []
        labels = []

        for class_idx in range(self.c.num_classes):
            for _ in range(num_samples_per_class):
                seq_len_i = torch.randint(
                    low=self.c.min_seq_len,
                    high=self.c.max_seq_len + 1,
                    size=(1,)
                ).item()

                seqs.append(self._generate_sample(class_idx, seq_len_i))
                labels.append(class_idx)

        labels = torch.tensor(labels, dtype=torch.long)
        return seqs, labels

    def _generate_sample(self, class_idx: int, seq_len_i: int) -> torch.Tensor:
        c = self.c
        seq = torch.randn(seq_len_i, c.input_dim) * c.noise_std

        # 1.py와 동일한 "클래스별 신호 주입 로직"을 seq_len_i에 맞춰 적용
        if class_idx == 0:
            seq[:, 0] = torch.randn(seq_len_i) * c.signal_noise + c.signal_strength
            seq[:, 1] = torch.randn(seq_len_i) * c.noise_std + 0.1
            seq[:, 2] = torch.randn(seq_len_i) * c.noise_std - 0.2
        elif class_idx == 1:
            seq[:, 0] = torch.randn(seq_len_i) * c.noise_std - 0.2
            seq[:, 1] = torch.randn(seq_len_i) * c.signal_noise + c.signal_strength
            seq[:, 2] = torch.randn(seq_len_i) * c.noise_std + 0.1
        else:
            seq[:, 0] = torch.randn(seq_len_i) * c.noise_std + 0.1
            seq[:, 1] = torch.randn(seq_len_i) * c.noise_std - 0.2
            seq[:, 2] = torch.randn(seq_len_i) * c.signal_noise + c.signal_strength

        seq = seq + torch.randn(seq_len_i, c.input_dim) * c.additional_noise
        return seq


# ============================================
# collate_fn: pad + key_padding_mask 생성
# - padded_x: (B, Lmax, D)
# - key_padding_mask: (B, Lmax), True인 곳이 padding (MultiheadAttention 규약)
# ============================================
def pad_collate_fn(batch, pad_value: float = 0.0):
    sequences, labels = zip(*batch)  # sequences: list[Tensor(L_i, D)]
    labels = torch.stack([torch.as_tensor(y) for y in labels]).long()

    lengths = torch.tensor([s.size(0) for s in sequences], dtype=torch.long)
    max_len = lengths.max().item()
    input_dim = sequences[0].size(1)

    padded_x = torch.full((len(sequences), max_len, input_dim), pad_value, dtype=sequences[0].dtype)

    # True = padding
    key_padding_mask = torch.ones((len(sequences), max_len), dtype=torch.bool)

    for i, s in enumerate(sequences):
        L = s.size(0)
        padded_x[i, :L] = s
        key_padding_mask[i, :L] = False

    return padded_x, labels, key_padding_mask, lengths


# ============================================
# 마스크를 받는 모델: base.SimpleAttentionModel을 "확장"해서 사용
# - 핵심: self_attention에 key_padding_mask 전달
# - pooling은 padding 제외 masked mean
# ============================================
class SimpleAttentionModelVarLen(base.SimpleAttentionModel):
    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        # x: (B, L, D)
        x = self.input_projection(x)

        attn_output, attn_weights = self.self_attention(
            x, x, x,
            key_padding_mask=key_padding_mask  # (B, L) True=pad
        )
        x = self.norm1(x + attn_output)

        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        # masked mean pooling (padding 제외)
        if key_padding_mask is None:
            pooled = x.mean(dim=1)
        else:
            valid_mask = (~key_padding_mask).unsqueeze(-1)  # (B, L, 1), True=valid
            x_sum = (x * valid_mask).sum(dim=1)             # (B, E)
            denom = valid_mask.sum(dim=1).clamp(min=1)      # (B, 1)
            pooled = x_sum / denom

        logits = self.classifier(pooled)
        return logits, attn_weights


# ============================================
# Trainer를 "최소 변경"해서 마스크를 같이 넘기도록 래핑
# - base.Trainer 로직을 최대한 재사용하되,
#   train/eval 루프에서 batch가 (x,y) 대신 (x,y,mask,lengths)로 들어옴
# ============================================
class TrainerVarLen(base.Trainer):
    def train_epoch(self, train_loader: DataLoader):
        self.model.train()

        epoch_loss = 0.0
        correct = 0
        total = 0

        for x, y, key_padding_mask, _lengths in train_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            key_padding_mask = key_padding_mask.to(self.device)

            logits, _ = self.model(x, key_padding_mask=key_padding_mask)
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
            for x, y, key_padding_mask, _lengths in test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                key_padding_mask = key_padding_mask.to(self.device)

                logits, _ = self.model(x, key_padding_mask=key_padding_mask)
                loss = self.criterion(logits, y)

                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        avg_loss = total_loss / len(test_loader)
        acc = correct / total * 100
        return avg_loss, acc


# ============================================
# Evaluator도 최소 변경 (mask 포함 배치 처리)
# ============================================
class EvaluatorVarLen(base.Evaluator):
    @staticmethod
    def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device, num_classes: int = 3):
        print(f"\n{'=' * 60}")
        print("클래스별 성능 분석")
        print(f"{'=' * 60}")

        model.eval()
        preds, trues = [], []

        with torch.no_grad():
            for x, y, key_padding_mask, _lengths in test_loader:
                x = x.to(device)
                key_padding_mask = key_padding_mask.to(device)
                logits, _ = model(x, key_padding_mask=key_padding_mask)

                preds.append(logits.argmax(dim=1).cpu())
                trues.append(y)

        preds = torch.cat(preds)
        trues = torch.cat(trues)

        cm = torch.zeros(num_classes, num_classes, dtype=torch.int)
        for t, p in zip(trues, preds):
            cm[t, p] += 1

        # (출력 형식은 1.py와 동일하게 유지)
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
# DataLoader 생성 (collate_fn만 추가)
# ============================================
def create_dataloaders(train_dataset, test_dataset, config: Config):
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle_train,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=pad_collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=pad_collate_fn
    )

    print(f"\n[DataLoader 정보]")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Batch size: {config.batch_size}")

    return train_loader, test_loader


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
    gen = DataGeneratorVarLen(config)
    train_dataset = gen.generate_dataset(config.num_train_samples_per_class)
    test_dataset = gen.generate_dataset(config.num_test_samples_per_class)

    base.print_all_dataset_samples(train_dataset, "train")
    base.print_all_dataset_samples(test_dataset, "test")

    # 데이터 분석 (가변 길이라서 1.py 분석기를 그대로 쓰기 어렵기 때문에,
    # "설명 출력 유지" 취지로 최소한의 정보만 동일 톤으로 출력)
    print("\n[데이터셋 정보]")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    sample_x, sample_y = train_dataset[0]
    print(f"Sample data shape: {sample_x.shape}")
    print(f"Sample label type: {type(int(sample_y))}")

    # 클래스 분포는 동일하게 출력 가능
    # labels 텐서 형태가 train_dataset.labels 가 아니라서 아래처럼 구현
    train_labels = train_dataset.labels
    test_labels = test_dataset.labels

    def print_class_dist(labels, name):
        num_classes = labels.max().item() + 1
        print(f"\n[{name} 클래스 분포]")
        for i in range(num_classes):
            count = (labels == i).sum().item()
            pct = count / len(labels) * 100
            print(f"  Class {i}: {count} samples ({pct:.1f}%)")

    print_class_dist(train_labels, "Train")
    print_class_dist(test_labels, "Test")

    # DataLoader
    train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, config)

    # 디바이스
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Device]: {device}")

    # 모델
    model = SimpleAttentionModelVarLen(config).to(device)

    # torchinfo summary는 가변 길이를 직접 요약하기 어려워서 대표 길이(max_seq_len)로 출력
    # (학습은 실제로 batch마다 pad 길이 달라짐)
    summary(model, input_size=(config.batch_size, config.max_seq_len, config.input_dim))

    # 학습
    trainer = TrainerVarLen(model, config, device)
    trainer.train(train_loader, test_loader)

    # 최종 결과 출력
    print(f"\n{'=' * 60}")
    print("학습 완료!")
    print(f"{'=' * 60}")
    print(f"최종 Train Accuracy: {trainer.train_accuracies[-1]:.2f}%")
    print(f"최종 Test Accuracy: {trainer.test_accuracies[-1]:.2f}%")
    print(f"최고 Test Accuracy: {trainer.best_test_acc:.2f}%")
    print(f"최종 Learning Rate: {trainer.learning_rates[-1]:.6f}")

    # 시각화는 base.Visualizer 그대로 사용 가능하지만,
    # attention heatmap은 pad 포함된 Lmax 기준으로 나오니 "대표 샘플"로 보는 용도로는 OK
    base.Visualizer.plot_training_results(trainer, _WrappedTestDatasetForViz(test_dataset, config), device)

    # 평가
    EvaluatorVarLen.evaluate_model(model, test_loader, device, config.num_classes)

    print(f"\n{'=' * 60}")
    print("✅ 학습 및 분석 완료!")
    print(f"{'=' * 60}")


# ============================================
# (선택) base.Visualizer가 test_dataset[0]이 (Tensor, label) 형태이길 기대해서 래핑
# - base.Visualizer는 내부에서 test_dataset[0]을 꺼내 sample_data.unsqueeze(0) 해서 모델에 넣음
# - 여기서는 VariableLength라 pad가 없으니, max_seq_len까지 pad해주고 "mask 없이" 넣게끔 래핑
#   (시각화용이라 이렇게 간단 처리)
# ============================================
class _WrappedTestDatasetForViz(torch.utils.data.Dataset):
    def __init__(self, varlen_dataset: VariableLengthSequenceDataset, config: Config):
        self.ds = varlen_dataset
        self.c = config

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]
        L = x.size(0)
        if L < self.c.max_seq_len:
            pad = torch.zeros(self.c.max_seq_len - L, self.c.input_dim, dtype=x.dtype)
            x = torch.cat([x, pad], dim=0)
        else:
            x = x[: self.c.max_seq_len]
        return x, y


if __name__ == "__main__":
    main()