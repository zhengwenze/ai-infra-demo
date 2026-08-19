"""
Day 9: Hugging Face Tokenizer 最小可运行实验
=============================================
学习目标：
1. 理解 Tokenizer 是什么：把文本 → Token ID 序列的工具
2. 使用真实的 Qwen2.5 tokenizer 进行中文/英文 tokenize
3. 理解 input_ids、decode、特殊 token 的概念
4. 直观感受「一句话被切成多少个 token」
"""

from transformers import AutoTokenizer


def print_separator(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


print_separator("1. 加载 Qwen2.5-0.5B-Instruct Tokenizer")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

print(f"Tokenizer 类型: {type(tokenizer).__name__}")
print(f"词表大小 (vocab_size): {tokenizer.vocab_size}")
print(f"特殊 token: {tokenizer.special_tokens_map}")


# ============================================================
# 2. 中文 Tokenize：第一个实验
# ============================================================
print_separator("2. 中文 Tokenize 实验")

text = "我正在学习大模型推理"
print(f"原始文本: {text}")
print(f"字符数: {len(text)}")

# tokenize：文本 → token id 序列
encoded = tokenizer(text, add_special_tokens=False)
print(f"\nencoded 完整结果: {encoded}")

input_ids = encoded["input_ids"]
print(f"\ninput_ids = {input_ids}")
print(f"token 数量: {len(input_ids)}")

# decode：token id 序列 → 文本（还原）
decoded = tokenizer.decode(input_ids)
print(f"\ndecode 还原: {decoded}")

# 逐个 token 查看
print(f"\n逐个 Token 对应：")
for i, tid in enumerate(input_ids):
    token_str = tokenizer.decode([tid])
    print(f"  [{i}] ID={tid:<6d}  →  '{token_str}'")


# ============================================================
# 3. 中英混合 Tokenize 对比
# ============================================================
print_separator("3. 中英混合 Tokenize 对比")

texts = [
    "我正在学习大模型推理",
    "I am learning LLM inference",
    "大模型 Inference 推理 2025",
    "Hello, 世界！",
]

print(f"{'文本':<30s} {'字符数':<8s} {'Token数':<8s} {'压缩比(字符/Token)':<15s}")
print(f"  {'─'*70}")
for t in texts:
    ids = tokenizer(t, add_special_tokens=False)["input_ids"]
    n_chars = len(t)
    n_tokens = len(ids)
    ratio = n_chars / max(n_tokens, 1)
    print(f"  {t:<28s} {n_chars:<8d} {n_tokens:<8d} {ratio:<15.2f}")

print(f"\n观察：")
print(f"  • 中文：约 1 个汉字 ≈ 1 个 token（有时 1 汉字 = 多 token）")
print(f"  • 英文：1 个单词 ≈ 1~2 个 token")
print(f"  • 数字/标点：可能独立成 token 或合并")


# ============================================================
# 4. add_special_tokens 的区别
# ============================================================
print_separator("4. add_special_tokens 对比")

text = "你好"
print(f"文本: '{text}'")

without_special = tokenizer(text, add_special_tokens=False)
with_special = tokenizer(text, add_special_tokens=True)

print(f"  无 special tokens: {without_special['input_ids']}")
print(f"  有 special tokens: {with_special['input_ids']}")

# 打印已知特殊 token ID
print(f"\n  bos_token: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")
print(f"  eos_token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
print(f"  pad_token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
print(f"\n  说明: Qwen2.5 默认无 bos_token，eos_token=<|im_end|> 用于标记对话结束")


# ============================================================
# 5. batch tokenize：多条文本一次性编码
# ============================================================
print_separator("5. Batch Tokenize（批量编码 + padding）")

texts_batch = ["大模型推理", "PyTorch"]

# 不 padding
batch_no_pad = tokenizer(texts_batch, add_special_tokens=False)
print(f"无 padding:")
for i, (t, ids) in enumerate(zip(texts_batch, batch_no_pad["input_ids"])):
    print(f"  '{t}' → {ids} (len={len(ids)})")

# 有 padding（对齐到最长）
batch_padded = tokenizer(texts_batch, add_special_tokens=False, padding=True)
print(f"\n有 padding (对齐到最长):")
for i, (t, ids) in enumerate(zip(texts_batch, batch_padded["input_ids"])):
    print(f"  '{t}' → {ids} (len={len(ids)})")
print(f"  attention_mask: {batch_padded['attention_mask']}")
print(f"  说明: attention_mask=0 的位置是 padding，模型推理时应忽略")


# ============================================================
# 6. 验收标准要点
# ============================================================
print_separator("6. 验收标准要点")

points = [
    (
        "Tokenizer 的作用",
        "把文本转换为 Token ID 序列（input_ids），这是大模型能理解的「数字」形式",
    ),
    ("input_ids 是什么", "一段文字对应的 Token ID 序列，模型实际处理的就是这些整数"),
    (
        "为什么需要 add_special_tokens=False",
        "学习阶段只看文本本身的 token；加上后会插入 bos/eos 等对话标记，干扰观察",
    ),
    (
        "中文和英文的 token 效率",
        "中文约 1 字 ≈ 1 token；英文约 1 词 ≈ 1~2 token。词表越大，压缩率越高",
    ),
    ("decode 的作用", "把 Token ID 序列还原回文本，验证 tokenize 正确性"),
]
for q, a in points:
    print(f"\n  Q: {q}")
    print(f"     {a}")

print("\n" + "=" * 65)
print("✅ tokenize_experiment.py 运行完成！")
print("=" * 65)
