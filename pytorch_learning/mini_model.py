"""
Day 8: 小模型与推理模式（inference_mode）实验
===============================================
学习目标：
1. 用 torch.nn.Sequential 快速搭建一个小型全连接网络
2. 理解 [batch_size, feature_dim] 的含义（32, 128）
3. 掌握 torch.inference_mode() 的用法与优势
4. 理解『训练』与『推理』的区别
"""

import time
import torch
import torch.nn as nn


def print_separator(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


print_separator("0. 设备选择")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    device_name = "MPS (Apple GPU)"
elif torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
else:
    device = torch.device("cpu")
    device_name = "CPU"

print(f"当前设备: {device_name}")


# ============================================================
# 1. 搭建小模型：nn.Sequential
# ============================================================
print_separator("1. 搭建小模型 (torch.nn.Sequential)")

model = nn.Sequential(
    # 第 1 层：Linear(128 → 256)
    #   输入特征维度 128，输出特征维度 256
    #   参数：权重 W1 (256, 128) + 偏置 b1 (256,)
    nn.Linear(in_features=128, out_features=256),
    nn.ReLU(),  # 激活函数：逐元素 max(x, 0)，引入非线性
    # 第 2 层：Linear(256 → 64)
    #   输入特征维度 256（必须等于上一层的输出维度），输出 64
    #   参数：权重 W2 (64, 256) + 偏置 b2 (64,)
    nn.Linear(in_features=256, out_features=64),
)

# 把模型搬到目标设备
model = model.to(device)

print(f"模型结构:\n{model}")
print(f"\n模型参数一览：")
total_params = 0
for name, param in model.named_parameters():
    n = param.numel()
    total_params += n
    print(
        f"  {name:<30s} shape={str(param.shape):<20s} dtype={param.dtype}  device={param.device}  参数数={n:,}"
    )
print(f"  {'─'*60}")
print(f"  总参数量: {total_params:,} 个")

# 这就是为什么参数必须有 dtype 和 device：
#   dtype  → 决定参数精度/体积（float32 每个占 4 字节，float16 占 2 字节）
#   device → 决定参数存在哪里（CPU 内存 or GPU 显存），
#            前向计算时输入 x 必须和模型参数在同一个 device 上！


# ============================================================
# 2. 准备输入：理解 [32, 128] 的含义
# ============================================================
print_separator("2. 准备输入 x，shape = [32, 128]")

batch_size = 32
feature_dim = 128
x = torch.randn(batch_size, feature_dim, device=device)

print(f"x.shape = {x.shape}")
print(f"  第 0 维 (32) : batch_size，一次前向同时处理『32 个样本』")
print(f"  第 1 维 (128): 每个样本的『特征维度』，即每个样本用 128 个 float 描述")
print(f"  类比: 32 张图片，每张图被展平后有 128 个像素/特征")


# ============================================================
# 3. 普通模式（默认开启梯度追踪 vs inference_mode）对比
# ============================================================
print_separator("3. 训练模式 vs 推理模式 (inference_mode)")

# --- 3a. 普通模式（默认，追踪梯度，适合训练）---
print("\n[模式 A] 默认模式（requires_grad 打开）—— 用于训练")
print(f"  model.training = {model.training}  (model.train())")
model.train()  # 切换到训练模式（影响 BatchNorm/Dropout 等，本例没用到）

# 检查参数的梯度追踪
for name, p in list(model.named_parameters())[:1]:
    print(f"  {name}.requires_grad = {p.requires_grad}  (默认 True，会保存计算图)")

t0 = time.time()
for _ in range(1000):
    output_train = model(x)
t_train = time.time() - t0

print(f"  output.shape = {output_train.shape} = [batch=32, out_features=64]")
print(f"  output.requires_grad = {output_train.requires_grad}  (因为参数追踪梯度)")
print(f"  1000 次前向耗时: {t_train*1000:.2f} ms")
print(f"  说明: 这种模式会为反向传播构建『计算图』，占用额外显存，适用于训练阶段。")


# --- 3b. inference_mode 模式（关闭梯度，适合推理） ---
print("\n[模式 B] torch.inference_mode() —— 用于推理/预测")
model.eval()  # 切换到评估模式（影响 BatchNorm/Dropout 等）

with torch.inference_mode():
    # 在这个上下文内：
    #   • 所有操作都不会跟踪梯度
    #   • 不会构建计算图
    #   • 显存和速度都会更优
    t0 = time.time()
    for _ in range(1000):
        output_infer = model(x)
    t_infer = time.time() - t0

print(f"  output.shape = {output_infer.shape} = 仍然 [32, 64]")
print(f"  output.requires_grad = {output_infer.requires_grad}  (已关闭)")
print(f"  1000 次前向耗时: {t_infer*1000:.2f} ms")
speedup = t_train / max(t_infer, 1e-9)
print(f"  相对普通模式速度提升: {speedup:.2f}×")
print(f"  说明: 推理/预测阶段我们只需要输出结果，不需要更新参数，")
print(f"        所以不需要保存反向传播信息（计算图、中间激活梯度）。")

# 数值结果一致
print(
    f"\n数值一致性检查 (两种模式输出是否一致): "
    f"{'✅ 一致' if torch.allclose(output_train, output_infer, atol=1e-6) else '❌ 有差异'}"
)


# ============================================================
# 4. 训练 vs 推理的区别总结
# ============================================================
print_separator("4. 训练 (train) 与 推理 (inference) 的区别")

comparison = [
    ("目标", "学参数：降低 loss，更新 W、b", "用学好的参数直接做预测"),
    ("前向", "需要计算输出 + loss", "只需要计算输出"),
    ("反向传播", "必须调用 loss.backward() 算梯度", "完全不需要"),
    ("参数更新", "optimizer.step() 更新参数", "参数固定不动"),
    (
        "梯度 / 计算图",
        "必须保留（耗显存和算力）",
        "可关闭 → inference_mode/no_grad 省显存提速",
    ),
    (
        "BatchNorm/Dropout",
        "使用当前 batch 统计，启用 dropout",
        "用全局统计量，关闭 dropout → model.eval()",
    ),
    ("典型场景", "训练集上迭代多轮 (epoch)", "验证/测试/上线服务 (单轮前向)"),
]
print(f"  {'项目':<20s}{'训练 (Training)':<40s}{'推理 (Inference)':<40s}")
print(f"  {'─'*95}")
for row in comparison:
    print(f"  {row[0]:<18s} {row[1]:<38s} {row[2]:<38s}")


# ============================================================
# 5. 手动模拟一次"训练"步骤（直观对比）
# ============================================================
print_separator("5. 直观对比：训练步骤（含 backward） vs 推理步骤")

# —— 训练风格：需要 grad / backward / optimizer ——
print("\n▶ 训练步骤示意（需保存梯度 & 反向传播）：")
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

x_train = torch.randn(32, 128, device=device)
y_target = torch.randn(32, 64, device=device)  # 假设的"真值"

pred = model(x_train)  # 1) 前向：生成计算图
loss = ((pred - y_target) ** 2).mean()  # 2) 算 loss
print(f"    loss = {loss.item():.4f}  (loss.requires_grad = {loss.requires_grad})")

# 下面三行是训练特有的，推理完全不需要
loss.backward()  # 3) 反向：从 loss 开始倒推梯度（需要计算图）
print(
    f"    backward() 完成：第 1 层权重梯度 norm = {model[0].weight.grad.norm().item():.4f}"
)
optimizer.step()  # 4) 更新：W ← W - lr * grad
optimizer.zero_grad()  # 5) 清零梯度，准备下一轮
print(f"    optimizer.step() 完成：参数已更新（训练才做）")

# —— 推理风格：inference_mode + model.eval() ——
print("\n▶ 推理步骤示意（关闭梯度，无 backward）：")
model.eval()
x_test = torch.randn(32, 128, device=device)
with torch.inference_mode():
    preds = model(x_test)  # 只需前向！没有 loss、没有 backward、没有 step
    print(f"    预测 output.shape = {preds.shape}")
    print(f"    preds.requires_grad = {preds.requires_grad}  (不存计算图)")
print(f"    搞定！这就是为什么推理不用保存反向传播信息——压根不用 backward/step。")


# ============================================================
# 6. 验收标准总结
# ============================================================
print_separator("6. Day 8 验收标准")

answers = [
    (
        "[32, 128] 的含义",
        "batch_size=32（一次处理 32 个样本）；feature_dim=128（每个样本用 128 个特征表示）",
    ),
    (
        "如何判断两矩阵能否相乘",
        "A.shape[1] 必须 == B.shape[0]，即『前一个的列数 = 后一个的行数』。结果的 shape = (A.shape[0], B.shape[1])",
    ),
    (
        "参数为什么有 dtype 和 device",
        "dtype 决定精度&显存占用（float32/fp16/bf16）；device 决定参数存在 CPU/GPU，前向时输入必须同设备。",
    ),
    (
        "推理为什么不需要保存反向传播信息",
        "推理只做前向预测，不会 backward → 不会算梯度 → 不需要构建计算图和保存中间激活梯度。"
        "关闭后能省大量显存+加速（inference_mode/no_grad）。",
    ),
]
for q, ans in answers:
    print(f"\n  Q: {q}")
    print(f"     {ans}")

print("\n" + "=" * 65)
print("✅ mini_model.py 运行完成！")
print("=" * 65)
