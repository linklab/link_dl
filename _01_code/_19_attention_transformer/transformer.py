import torch
import torch.nn as nn
import math

print("=" * 60)
print("Transformer 예제")
print("=" * 60)

# 파라미터 설정
src_seq_len = 12          # 소스 시퀀스 길이 (Tx)
tgt_seq_len = 10          # 타겟 시퀀스 길이 (Ty)
src_vocab_size = 1000     # 소스 어휘 크기 (예: 영어 어휘 사전의 단어 개수 - 500,000개)
tgt_vocab_size = 1500     # 타겟 어휘 크기 (예: 프랑스어 어휘 사전의 단어 개수 - 20,000개)
d_model = 512             # 임베딩 차원 (h)
nhead = 8                 # Multi-head attention의 head 개수 (d)
num_encoder_layers = 6    # Encoder 레이어 개수 (Nx)
num_decoder_layers = 6    # Decoder 레이어 개수 (Nx)
dim_feedforward = 2048    # Feed-Forward 네트워크의 은닉층 차원
batch_size = 2

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # 위치 인코딩 텐서 생성: (max_len, d_model)
        position_encoding = torch.zeros(max_len, d_model)

        # pos: 0, 1, 2, ..., max_len-1 형태의 (max_len, 1) 텐서
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # i: 0, 2, 4, ..., d_model-2 (짝수 인덱스만)
        # 2i/d_model 계산을 위한 텐서
        i = torch.arange(0, d_model, 2, dtype=torch.float)

        # 분모 계산: 10000^(2i/d_model)
        denominator = torch.pow(10000, i / d_model)

        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        position_encoding[:, 0::2] = torch.sin(pos / denominator)

        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        position_encoding[:, 1::2] = torch.cos(pos / denominator)

        # (max_len, d_model) -> (1, max_len, d_model)로 배치 차원 추가
        position_encoding = position_encoding.unsqueeze(0)

        # 학습되지 않는 버퍼로 등록
        self.register_buffer('position_encoding', position_encoding)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        # 시퀀스 길이만큼만 위치 인코딩 추가
        return x + self.position_encoding[:, :x.size(1), :]


class TransformerModel(nn.Module):
    """
    Transformer 모델
    - Encoder: Self-Attention + Feed-Forward
    - Decoder: Masked Self-Attention + Cross-Attention + Feed-Forward
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead,
                 num_encoder_layers, num_decoder_layers, dim_feedforward):
        super().__init__()

        self.d_model = d_model

        # ============================================
        # 1. Embedding Layers
        # ============================================
        self.src_embedding = nn.Embedding(num_embeddings=src_vocab_size, embedding_dim=d_model)
        self.tgt_embedding = nn.Embedding(num_embeddings=tgt_vocab_size, embedding_dim=d_model)

        # ============================================
        # 2. Positional Encoding
        # ============================================
        self.pos_encoder = PositionalEncoding(d_model)

        # ============================================
        # 3. Transformer (Encoder + Decoder)
        # ============================================
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True  # (batch, seq, feature) 형태 사용
        )

        # ============================================
        # 4. Output Projection (Linear + Softmax)
        # ============================================
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        Args:
            src: 소스 시퀀스 (batch, src_seq_len) - 정수 인덱스
            tgt: 타겟 시퀀스 (batch, tgt_seq_len) - 정수 인덱스
            src_key_padding_mask: 소스 패딩 마스크 (batch, src_seq_len) - bool
            tgt_key_padding_mask: 타겟 패딩 마스크 (batch, tgt_seq_len) - bool

        Returns:
            logits: (batch, tgt_seq_len, tgt_vocab_size)
        """

        # ============================================
        # Step 1: Embedding + Scaling + Positional Encoding
        # ============================================
        # sqrt(d_model)로 스케일링 (원논문)
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model) # (batch, src_seq_len, d_model)
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model) # (batch, src_seq_len, d_model)

        # Positional Encoding 추가
        src_embedded = self.pos_encoder(src_embedded)  # (batch, src_seq_len, d_model)
        tgt_embedded = self.pos_encoder(tgt_embedded)  # (batch, tgt_seq_len, d_model)

        # ============================================
        # Step 2: Causal Mask 생성 (Decoder용)
        # ============================================
        # 타겟 시퀀스에서 미래 토큰을 보지 못하게 마스킹
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            sz=tgt.size(1), device=tgt.device, dtype=torch.bool
        )

        # ============================================
        # Step 3: Transformer Forward
        # ============================================
        # Encoder: Self-Attention으로 소스 시퀀스 인코딩
        # Decoder: Masked Self-Attention + Cross-Attention으로 타겟 생성
        transformer_output = self.transformer(
            src=src_embedded,                               # Encoder 입력
            tgt=tgt_embedded,                               # Decoder 입력
            tgt_mask=tgt_mask,                              # Causal mask
            src_key_padding_mask=src_key_padding_mask,      # 소스 패딩 마스크
            tgt_key_padding_mask=tgt_key_padding_mask,      # 타겟 패딩 마스크
            memory_key_padding_mask=src_key_padding_mask    # Cross-Attention용 소스 패딩
        )

        # ============================================
        # Step 4: Output Projection
        # ============================================
        # (batch, tgt_seq_len, d_model) -> (batch, tgt_seq_len, tgt_vocab_size)
        logits = self.output_projection(transformer_output)

        return logits

def pad_batch(sequences, pad_value=0):
    """
    sequences: 길이가 제각각인 토큰 시퀀스의 리스트 (list of list[int])
              예) [[5, 10, 11], [7, 3], [9, 8, 1, 4]]
    pad_value: 패딩에 사용할 토큰 인덱스 (기본 0)

    Returns:
        padded: (batch, max_len) LongTensor
        padding_mask: (batch, max_len) BoolTensor
                      True  = padding 위치 (무시)
                      False = 실제 토큰 위치
    """
    batch_size = len(sequences)
    max_len = max(len(seq) for seq in sequences)

    # 먼저 전부 pad_value로 채워놓고
    padded = torch.full((batch_size, max_len), pad_value, dtype=torch.long)
    # 처음엔 전부 padding(True)라고 가정했다가 실제 토큰 있는 부분만 False로 바꿔줌
    padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)

    for i, seq in enumerate(sequences):
        length = len(seq)
        padded[i, :length] = torch.tensor(seq, dtype=torch.long)
        padding_mask[i, :length] = False  # 실제 토큰이 있는 부분은 패딩이 아님(False)

    return padded, padding_mask

# ============================================
# 모델 생성
# ============================================
model = TransformerModel(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=d_model,
    nhead=nhead,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    dim_feedforward=dim_feedforward
)

print(f"\n✅ 모델 생성 완료")
print(f"   - 총 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

# ============================================
# Case 1: 패딩 마스크 없이 실행 (고정 길이 배치)
# ============================================
print("\n" + "=" * 60)
print("Case 1: 패딩 마스크 없이 실행 (고정 길이)")
print("=" * 60)

# 소스 예: "I am a student" -> 정수 인덱스로 변환된 형태 (단어 개수: src_seq_len)
src_case1 = torch.randint(low=0, high=src_vocab_size, size=(batch_size, src_seq_len))

# 타겟 예: "<sos> je suis étudiant" -> 정수 인덱스로 변환된 형태 (단어 개수: tgt_seq_len)
tgt_case1 = torch.randint(low=0, high=tgt_vocab_size, size=(batch_size, tgt_seq_len))

print(f"[입력 데이터]")
print(f"   - src_case1 shape: {src_case1.shape} (batch, Tx)")
print(f"   - tgt_case1 shape: {tgt_case1.shape} (batch, Ty)")

output_case1 = model(src_case1, tgt_case1)

print(f"\n[출력 데이터]")
print(f"   - output_case1 shape: {output_case1.shape}")
print(f"   - (batch, Ty, tgt_vocab_size)")
print(f"   - 각 타겟 토큰마다 어휘 전체에 대한 확률 분포 생성")


# ============================================
# Case 2: 패딩 마스크 포함 (고정 길이 배치, 일부 토큰이 패딩)
# ============================================
print("\n" + "=" * 60)
print("Case 2: 패딩 마스크와 함께 실행 (고정 길이, 일부 패딩)")
print("=" * 60)

# 입력 데이터 (Case 1과 동일한 크기)
src_case2 = torch.randint(low=0, high=src_vocab_size, size=(batch_size, src_seq_len))
tgt_case2 = torch.randint(low=0, high=tgt_vocab_size, size=(batch_size, tgt_seq_len))

# 패딩 마스크 생성 (True = 패딩 위치, attention에서 무시)
src_key_padding_mask_case2 = torch.zeros(batch_size, src_seq_len, dtype=torch.bool)
tgt_key_padding_mask_case2 = torch.zeros(batch_size, tgt_seq_len, dtype=torch.bool)

# 마지막 2개 토큰을 패딩으로 설정
src_key_padding_mask_case2[:, -2:] = True
tgt_key_padding_mask_case2[:, -2:] = True

print(f"[입력 데이터]")
print(f"   - src_case2 shape: {src_case2.shape} (batch, Tx)")
print(f"   - tgt_case2 shape: {tgt_case2.shape} (batch, Ty)")
print(f"\n[패딩 마스크]")
print(f"   - src_key_padding_mask_case2[0]: {src_key_padding_mask_case2[0]}")
print(f"   - tgt_key_padding_mask_case2[0]: {tgt_key_padding_mask_case2[0]}")

output_case2 = model(src_case2, tgt_case2, src_key_padding_mask_case2, tgt_key_padding_mask_case2)

print(f"\n[출력 데이터]")
print(f"   - output_case2 shape: {output_case2.shape}")
print(f"   - 패딩 토큰은 attention 계산에서 무시됨")


# ============================================
# Case 3: 서로 길이 다른 문장들이 섞여 있는 배치 처리 (가변 길이)
# ============================================
print("\n" + "=" * 60)
print("Case 3: 가변 길이 시퀀스를 pad + mask로 한 번에 처리")
print("=" * 60)

# 예시용 source/target 시퀀스들 (길이가 다름)
# 실제로는 여기에 토크나이저로 변환된 인덱스 시퀀스를 넣게 됨
src_sentences_case3 = [
    [5, 27, 99, 13],           # 길이 4
    [6, 3, 8, 12, 45, 9, 2]    # 길이 7
]

tgt_sentences_case3 = [
    [1, 4, 5, 6],              # 길이 4 (예: <sos> 토큰 포함)
    [1, 7, 8]                  # 길이 3
]

# pad_batch를 이용해 텐서와 padding mask 생성
src_case3, src_key_padding_mask_case3 = pad_batch(src_sentences_case3, pad_value=0)
tgt_case3, tgt_key_padding_mask_case3 = pad_batch(tgt_sentences_case3, pad_value=0)

print("[입력 데이터]")
print(f"   - src_case3 shape: {src_case3.shape} (batch, Tx_max)")
print(f"   - tgt_case3 shape: {tgt_case3.shape} (batch, Ty_max)")
print(f"\n   - src_case3[0] (첫 번째 소스 문장): {src_case3[0]}")
print(f"   - src_case3[1] (두 번째 소스 문장): {src_case3[1]}")
print(f"\n   - tgt_case3[0] (첫 번째 타겟 문장): {tgt_case3[0]}")
print(f"   - tgt_case3[1] (두 번째 타겟 문장): {tgt_case3[1]}")
print(f"\n[패딩 마스크]")
print(f"   - src_key_padding_mask_case3[0]: {src_key_padding_mask_case3[0]}")
print(f"   - src_key_padding_mask_case3[1]: {src_key_padding_mask_case3[1]}")
print(f"\n   - tgt_key_padding_mask_case3[0]: {tgt_key_padding_mask_case3[0]}")
print(f"   - tgt_key_padding_mask_case3[1]: {tgt_key_padding_mask_case3[1]}")

# 모델에 넣기
output_case3 = model(
    src_case3,
    tgt_case3,
    src_key_padding_mask=src_key_padding_mask_case3,
    tgt_key_padding_mask=tgt_key_padding_mask_case3
)

print("\n[출력 데이터]")
print(f"   - output_case3 shape: {output_case3.shape}")
print("   - padding인 위치는 attention에서 무시되며, 실제 토큰 위치만 사용됨")