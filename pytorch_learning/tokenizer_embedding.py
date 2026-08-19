"""
Day 9: Tokenizer、Embedding 与采样 — 完整链路实验
=====================================================
完整链路：
  文本 → Tokenizer → Token IDs → Embedding → Transformer → Logits → Sampling → 下一个 Token

核心概念（三个 ≠）：
  1. 字符数 ≠ Token 数
  2. Token ID ≠ Token 向量
  3. Logits ≠ 概率

验收标准：
  - 能解释 Tokenizer 的作用
  - 能解释 vocabulary
  - 能解释 Embedding 为什么是一个查表过程
  - 能说明 temperature 变大后输出为什么更随机
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer


def print_separator(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


# ============================================================
# 0. 设备选择
# ============================================================
print_separator("0. 设备选择")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"设备: MPS (Apple GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"设备: CUDA ({torch.cuda.get_device_name(0)})")
else:
    device = torch.device("cpu")
    print(f"设备: CPU")


# ============================================================
# 1. Tokenizer：文本 → Token IDs
# ============================================================
print_separator("1. Tokenizer：文本 → Token IDs")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
print(f"Tokenizer 类型: {type(tokenizer).__name__}")
print(f"词表大小 (vocab_size): {tokenizer.vocab_size}")
print(f"  → vocabulary 就是这个「字典」：151643 个条目，每个条目 = 一个 token")
print(f"  → 每个 token 有唯一编号 (0 ~ 151642)，这就是 Token ID")

# 中文 tokenize
text_zh = "我正在学习大模型推理"
ids_zh = tokenizer(text_zh, add_special_tokens=False)["input_ids"]

# 英文 tokenize
text_en = "I am learning LLM inference"
ids_en = tokenizer(text_en, add_special_tokens=False)["input_ids"]

print(f"\n中文: '{text_zh}'")
print(f"  字符数={len(text_zh)}, Token数={len(ids_zh)}")
print(f"  input_ids = {ids_zh}")
print(f"  逐 token: {[tokenizer.decode([t]) for t in ids_zh]}")

print(f"\n英文: '{text_en}'")
print(f"  字符数={len(text_en)}, Token数={len(ids_en)}")
print(f"  input_ids = {ids_en}")
print(f"  逐 token: {[tokenizer.decode([t]) for t in ids_en]}")

print(f"\n  ★ 字符数 ≠ Token 数：")
print(f"    中文 10 字 → {len(ids_zh)} tokens；英文 {len(text_en)} 字 → {len(ids_en)} tokens")

# decode 还原
decoded_zh = tokenizer.decode(ids_zh)
decoded_en = tokenizer.decode(ids_en)
print(f"\n  decode 还原: 中文 '{decoded_zh}', 英文 '{decoded_en}'")


# ============================================================
# 2. Embedding：Token IDs → 向量（查表过程）
# ============================================================
print_separator("2. Embedding：Token IDs → 向量（查表过程）")

vocab_size = tokenizer.vocab_size  # 151643
embed_dim = 256  # 每个 token 用 256 维向量表示（真实模型通常 1024~4096）

# nn.Embedding 本质上就是一个 [vocab_size, embed_dim] 的矩阵
embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim).to(device)

print(f"Embedding 矩阵 shape: {embedding.weight.shape}  # [词表大小, 向量维度]")
print(f"  → 本质就是一个大表：{vocab_size} 行 × {embed_dim} 列")
print(f"  → 每一行就是一个 token 对应的向量")

# 把中文 token_ids 转成向量
ids_tensor = torch.tensor(ids_zh, device=device)
embedded = embedding(ids_tensor)

print(f"\nToken IDs:     {ids_zh}")
print(f"  → shape: {ids_tensor.shape}  (一维，6个整数)")
print(f"Embedding 后:  shape={embedded.shape}  (6个token，每个256维向量)")
print(f"\n  ★ Token ID ≠ Token 向量：")
print(f"    ID 是一个整数（如 35946）；向量是 256 个浮点数（如 [0.12, -0.34, ...]）")
print(f"    Embedding 做的就是：拿 ID 当行号，从表里查出对应的那一行向量")

# 直观展示
print(f"\n查表示例：")
for i in range(3):
    print(f"  ID={ids_zh[i]:<6d} (token='{tokenizer.decode([ids_zh[i]])}') → "
          f"向量前5维: {embedded[i, :5].tolist()}")


# ============================================================
# 3. 完整链路：Embedding → Transformer → Logits
# ============================================================
print_separator("3. 完整链路：Embedding → 简单Transformer → Logits")

# 构建一个迷你 Transformer 模型（仅用于演示，不训练）
d_model = embed_dim  # 256
nhead = 4
num_layers = 2

class MiniGPT(nn.Module):
    """最小可运行的 Transformer 语言模型"""
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Transformer Decoder：能做 causal attention（因果注意力）
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            batch_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # 输出层：把 hidden state 映射回词表维度 → logits
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        # 1. Token IDs → Embedding 向量
        x = self.embedding(input_ids)  # [batch, seq_len, d_model]
        # 2. 构造 causal mask（下三角，让每个位置只能看到前面的 token）
        seq_len = input_ids.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        # 3. Transformer 处理（用全零 memory 作为 encoder 输出的占位）
        memory = torch.zeros(input_ids.shape[0], seq_len, d_model, device=x.device)
        x = self.transformer(x, memory, tgt_mask=mask)  # [batch, seq_len, d_model]
        # 4. 映射回词表维度 → logits
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]
        return logits


model = MiniGPT(vocab_size, d_model, nhead, num_layers).to(device)
model.eval()

# 用中文输入跑前向
input_ids = torch.tensor([ids_zh], device=device)  # [1, 6]
print(f"输入: '{text_zh}'")
print(f"  input_ids shape: {input_ids.shape}  # [batch=1, seq_len=6]")

with torch.inference_mode():
    logits = model(input_ids)  # [1, 6, vocab_size]

print(f"  logits shape: {logits.shape}  # [batch=1, seq=6, vocab=151643]")
print(f"\n  ★ Logits ≠ 概率：")
print(f"    logits 是原始得分（可正可负，未归一化）")
print(f"    需要经过 softmax 才变成概率（0~1，总和=1）")

# 看最后一个位置的 logits（预测下一个 token）
last_logits = logits[0, -1, :]  # [vocab_size]
print(f"\n最后一个位置的 logits 统计:")
print(f"  max={last_logits.max().item():.4f}, min={last_logits.min().item():.4f}")
print(f"  argmax (得分最高的 token ID) = {last_logits.argmax().item()}")
print(f"  对应 token = '{tokenizer.decode([last_logits.argmax().item()])}'")

# softmax 转概率
probs = F.softmax(last_logits, dim=-1)
print(f"\nsoftmax 后概率统计:")
print(f"  max prob={probs.max().item():.6f}, min prob={probs.min().item():.10f}")
print(f"  概率总和={probs.sum().item():.6f}  (必须=1.0)")


# ============================================================
# 4. 采样策略对比：greedy / temperature / top-k / top-p
# ============================================================
print_separator("4. 采样策略对比")

print(f"基于 logits 生成下一个 token，4 种策略对比：")
print(f"（每个策略采样 5 次，观察输出的多样性）\n")

# 统一使用最后一个位置的 logits
logits_for_sampling = last_logits.clone()

# --- 4a. Greedy（贪心）：永远选概率最大的 ---
print("【Greedy 贪心】→ argmax(logits)，每次结果相同")
for _ in range(5):
    next_id = logits_for_sampling.argmax().item()
    token = tokenizer.decode([next_id])
    print(f"  → ID={next_id}, token='{token}'")
print(f"  说明: 永远选最高分，确定性输出，没有随机性\n")

# --- 4b. Temperature（温度采样） ---
print("【Temperature 温度采样】→ logits / T 后 softmax 采样")
for T in [0.3, 1.0, 3.0]:
    torch.manual_seed(42)
    results = []
    for _ in range(5):
        scaled = logits_for_sampling / T
        probs_t = F.softmax(scaled, dim=-1)
        next_id = torch.multinomial(probs_t, num_samples=1).item()
        results.append(tokenizer.decode([next_id]))
    unique = len(set(results))
    print(f"  T={T:<4}: {results}  (去重后 {unique} 种)")
print(f"  说明: T→0 接近贪心(确定性)；T=1 正常；T→∞ 趋近均匀分布(完全随机)\n")

# --- 4c. Top-k（只从概率最高的 k 个中采样） ---
print("【Top-k 采样】→ 只保留 logits 最高的 k 个，其余设为 -inf")
for k in [1, 5, 50]:
    torch.manual_seed(42)
    results = []
    for _ in range(5):
        topk_vals, topk_idx = torch.topk(logits_for_sampling, k)
        masked = torch.full_like(logits_for_sampling, float('-inf'))
        masked[topk_idx] = topk_vals
        probs_k = F.softmax(masked, dim=-1)
        next_id = torch.multinomial(probs_k, num_samples=1).item()
        results.append(tokenizer.decode([next_id]))
    unique = len(set(results))
    print(f"  k={k:<3}: {results}  (去重后 {unique} 种)")
print(f"  说明: k=1 等于贪心；k 越大多样性越高\n")

# --- 4d. Top-p / Nucleus Sampling（累积概率达到 p 就截断） ---
print("【Top-p 核采样】→ 累积概率达 p 阈值的 token 集合中采样")
for p in [0.5, 0.9, 1.0]:
    torch.manual_seed(42)
    results = []
    for _ in range(5):
        sorted_probs, sorted_idx = torch.sort(F.softmax(logits_for_sampling, dim=-1), descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        # 找到累积概率超过 p 的位置，截断
        mask = cumsum <= p
        mask[..., 1:] = mask[..., 1:].clone() | (cumsum[..., :-1] < p)
        filtered = sorted_probs * mask.float()
        filtered_probs = filtered / filtered.sum()
        next_id = torch.multinomial(filtered_probs, num_samples=1).item()
        next_id = sorted_idx[next_id].item()
        results.append(tokenizer.decode([next_id]))
    unique = len(set(results))
    print(f"  p={p:<3}: {results}  (去重后 {unique} 种)")
print(f"  说明: p=1 等价普通采样；p 越小候选越少越确定\n")

# --- 总结对比表 ---
print("┌──────────────┬──────────────────────────────────────────┐")
print("│ 策略         │ 特点                                     │")
print("├──────────────┼──────────────────────────────────────────┤")
print("│ Greedy       │ 永远选最高分，完全确定，可能重复/保守    │")
print("│ Temperature  │ T↓=确定，T↑=随机；全局调节但可能选到烂词 │")
print("│ Top-k        │ 只从前 k 个选，固定候选数，简单高效      │")
print("│ Top-p        │ 动态候选数，概率集中时少选，分散时多选  │")
print("└──────────────┴──────────────────────────────────────────┘")


# ============================================================
# 5. temperature 变大为什么更随机：数学原理
# ============================================================
print_separator("5. temperature 变大为什么更随机？")

# 用一个简化的 3 token 例子
print("用 3 个 token 的例子直观说明：\n")
raw_logits = torch.tensor([2.0, 1.0, 0.5])

for T in [0.5, 1.0, 2.0, 10.0]:
    scaled = raw_logits / T
    probs = F.softmax(scaled, dim=-1)
    print(f"  T={T:<5} logits/T = {scaled.tolist()}")
    print(f"          probs     = {probs.tolist()}")
    print()

print("结论：")
print("  • T 变大 → logits 差距被缩小 → 概率分布趋于均匀")
print("  • 概率越均匀 → 每次采样的结果越不确定（更随机）")
print("  • 极端：T→∞ 时所有 token 概率相等 = 完全随机选")


# ============================================================
# 6. 验收标准总结
# ============================================================
print_separator("6. Day 9 验收标准")

answers = [
    ("Tokenizer 的作用",
     "把文本切分成 token 并映射为 Token ID 序列。是文字→数字的桥梁，模型只认数字。"),
    ("vocabulary 是什么",
     "词表 = 所有 token 的集合，每个 token 有唯一编号。Qwen2.5 词表有 151643 个 token。"),
    ("Embedding 为什么是查表过程",
     "nn.Embedding 内部是一个 [vocab_size, embed_dim] 矩阵。输入 token ID 当行号，直接取对应行向量。"
     "没有计算，就是查表（lookup table），和查字典一样。"),
    ("temperature 变大为什么更随机",
     "logits/T 后高分和低分的差距缩小，softmax 后概率分布趋于均匀。"
     "概率越均匀 → 每个 token 被选中的机会越接近 → 输出更不可预测（随机）。"),
    ("字符数 ≠ Token 数",
     "10 个中文字可能只有 6 个 token（高频词被合并），27 个英文字母 6 个 token（单词级切分）。"),
    ("Token ID ≠ Token 向量",
     "ID 是一个整数（如 35946），向量是 embed_dim 个浮点数（如 256 维）。Embedding 把 ID 映射为向量。"),
    ("Logits ≠ 概率",
     "Logits 是原始得分（任意实数），概率是 softmax 后的 [0,1] 且总和为 1。模型输出 logits，采样前要先转概率。"),
]
for q, a in answers:
    print(f"\n  Q: {q}")
    print(f"     {a}")

print("\n" + "="*65)
print("✅ tokenizer_embedding.py 运行完成！")
print("="*65)
