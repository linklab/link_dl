import torch
import torch.nn as nn

print("=" * 60)
print("Self-Attention 예제")
print("=" * 60)

# 파라미터 설정
seq_len = 10          # 시퀀스 길이 (Query, Key, Value 모두 동일)
input_dim = 256       # 입력 차원
embed_dim = 512       # Attention 내부 임베딩 차원
num_heads = 8         # Attention head 개수
batch_size = 2        # 배치 크기

print("\n💡 핵심:")
print("   - Self-Attention: Query, Key, Value가 모두 동일한 입력")
print("   - Query 입력 차원 = embed_dim 이어야 함")
print("   - 입력 차원이 embed_dim과 다르면 먼저 투영 필요\n")
print("-" * 60)

# 입력을 embed_dim으로 투영하는 레이어
input_projection = nn.Linear(input_dim, embed_dim)

# Self-Attention 생성
multi_heads_self_attention = nn.MultiheadAttention(
    embed_dim=embed_dim,      # Query, Key, Value의 차원 (모두 동일)
    num_heads=num_heads,
    batch_first=True
)

# 입력 생성 (하나의 시퀀스만 필요)
input_seq = torch.randn(batch_size, seq_len, input_dim)

print(f"[원본] 입력 shape: {input_seq.shape}")
print(f"  - batch_size: {batch_size}")
print(f"  - sequence_length: {seq_len}")
print(f"  - input_dim: {input_dim}")

# 입력을 embed_dim으로 투영
projected_input = input_projection(input_seq)
print(f"\n투영된 입력 shape: {projected_input.shape}")
print(f"  - embed_dim: {embed_dim}")

# Self-Attention 적용
# Query, Key, Value 모두 동일한 입력 사용
output, attn_weights = multi_heads_self_attention(
    projected_input,    # Query
    projected_input,    # Key (Query와 동일)
    projected_input,    # Value (Query와 동일)
)

print(f"\n[출력] shape: {output.shape}")
print(f"  - Query와 동일한 shape 유지")
print(f"\nAttention weights shape: {attn_weights.shape}")
print(f"  - ({batch_size}, {seq_len}, {seq_len})")
print(f"  - 각 토큰이 시퀀스의 모든 토큰에 대해 가지는 가중치")

# Attention weights 시각화
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=150)
print(f"\n{'=' * 60}")
print("Attention Weights 예시 (첫 번째 배치의 처음 10개 토큰)")
print(f"{'=' * 60}")
print(attn_weights[0, :10, :10].detach().numpy())
print("\n각 행: 해당 토큰이 다른 모든 토큰과 연관된 가중치")
print("각 행의 합 = 1.0 (softmax 결과)")
