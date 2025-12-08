import torch
import torch.nn as nn

print("=" * 60)
print("Cross-Attention 예제")
print("=" * 60)

print("⚠️  중요: nn.MultiheadAttention 에서는 Query의 입력 차원이 반드시 embed_dim과 일치해야 함!")
print("   kdim, vdim은 Key, Value의 입력 차원만 지정 가능")
print("-" * 60)

seq_len_q = 10
seq_len_kv = 15
query_dim = 256
key_value_dim = 384
embed_dim = 512
num_heads = 8
batch_size = 2

print("💡 핵심:")
print("   - Query 입력 차원 = embed_dim 이어야 함")
print("   - Key/Value 입력 차원 = kdim, vdim으로 지정 가능\n")

# Query를 먼저 투영
query_projection = nn.Linear(query_dim, embed_dim)

# Cross-Attention 생성
multi_heads_cross_attention = nn.MultiheadAttention(
    embed_dim=embed_dim,  # Query 입력 차원 (필수)
    num_heads=num_heads,
    kdim=key_value_dim,  # Key 입력 차원
    vdim=key_value_dim,  # Value 입력 차원
    batch_first=True
)

# 입력 생성
query = torch.randn(batch_size, seq_len_q, query_dim)
key = torch.randn(batch_size, seq_len_kv, key_value_dim)
value = torch.clone(key)

print(f"[원본] Query shape: {query.shape} (차원: {query_dim})")
print(f"Key shape: {key.shape} (차원: {key_value_dim})")
print(f"Value shape: {value.shape} (차원: {key_value_dim})")

# Query 투영
projected_query = query_projection(query)
print(f"\n투영된 Query shape: {projected_query.shape} (차원: {embed_dim})")

# Cross-Attention 적용
output, attn_weights = multi_heads_cross_attention(
    projected_query,    # 반드시 embed_dim 차원
    key,                # kdim으로 지정된 차원
    value,              # vdim으로 지정된 차원
)

print(f"\n[출력] Query shape: {output.shape}")
print(f"Attention weights shape: {attn_weights.shape}")
