"""
Day 8: PyTorch Tensor 基础实验
==============================
学习目标：
1. 理解 Tensor 是什么：PyTorch 的核心数据结构，类似于多维数组/矩阵
2. 掌握 shape、dtype、device 三个核心属性
3. 理解 CPU、MPS、CUDA 三种计算设备的区别
4. 掌握矩阵乘法的条件判断与计算
"""

import torch


def print_separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================
# 1. 基础环境检查
# ============================================================
print_separator("1. 基础环境检查")

print(f"PyTorch 版本: {torch.__version__}")
print(f"MPS (Apple Silicon GPU) 可用: {torch.backends.mps.is_available()}")
print(f"CUDA (NVIDIA GPU) 可用: {torch.cuda.is_available()}")

# 选择计算设备：优先 MPS -> CUDA -> CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    device_name = "Apple Silicon GPU (MPS)"
elif torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = f"NVIDIA GPU (CUDA) - {torch.cuda.get_device_name(0)}"
else:
    device = torch.device("cpu")
    device_name = "CPU"

print(f"当前使用设备: {device_name}")


# ============================================================
# 2. Tensor 是什么 & 核心属性 shape / dtype / device
# ============================================================
print_separator("2. Tensor 的核心属性：shape、dtype、device")

# 创建一个简单的 Tensor
a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

print(f"Tensor a:\n{a}")
print(f"  shape   : {a.shape}    # 表示维度：2行3列，即 (行数, 列数)")
print(f"  dtype   : {a.dtype}    # 数据类型，默认 float32")
print(f"  device  : {a.device}   # 存储/计算的设备，默认 cpu")

# 不同的 shape 示例
scalar = torch.tensor(3.14)
vec = torch.tensor([1.0, 2.0, 3.0])
mat = torch.randn(2, 3)
cube = torch.randn(4, 3, 2)

print(f"\nShape 维度示例：")
print(f"  标量 (0维) scalar.shape = {scalar.shape}")
print(f"  向量 (1维)   vec.shape = {vec.shape}")
print(f"  矩阵 (2维)   mat.shape = {mat.shape}")
print(f"  立方体 (3维) cube.shape = {cube.shape}")

# 不同的 dtype 示例
t_float32 = torch.tensor([1.0, 2.0])  # 默认
t_float64 = torch.tensor([1.0, 2.0], dtype=torch.float64)
t_int32 = torch.tensor([1, 2, 3], dtype=torch.int32)
t_int64 = torch.tensor([1, 2, 3], dtype=torch.int64)  # 常用于标签

print(f"\ndtype 类型示例：")
print(f"  float32 (默认) : {t_float32.dtype}, 参数占 4 字节/元素")
print(f"  float64 (双精) : {t_float64.dtype}, 参数占 8 字节/元素")
print(f"  int32          : {t_int32.dtype},   常用于整数运算")
print(f"  int64          : {t_int64.dtype},   常用于分类标签")


# ============================================================
# 3. CPU / MPS / CUDA 的区别：把 Tensor 迁移到不同设备
# ============================================================
print_separator("3. CPU / MPS / CUDA 三种设备")

x_cpu = torch.randn(2, 3)
print(f"默认创建: x_cpu.device = {x_cpu.device}")

# 迁移到目标设备（MPS/CUDA/CPU）
x_device = x_cpu.to(device)
print(f"迁移后    : x_device.device = {x_device.device}")

# 参数必须具有 dtype 和 device 的原因：
# - dtype: 决定精度和显存/内存占用，模型参数默认 float32
# - device: 决定参数存放在哪里（CPU内存 / GPU显存），
#           模型参数必须和输入数据在同一个 device 上才能计算
print(f"\n为什么参数需要 dtype 和 device？")
print(f"  • dtype  : 控制精度(float32/16)和显存占用，混合精度训练常用 float16")
print(f"  • device : 数据和模型必须放在同一设备上才能计算，否则报错")

# 演示设备不一致会报错
print(f"\n尝试 CPU Tensor 和 MPS Tensor 相加（会报错）：")
try:
    y = x_cpu + x_device
except RuntimeError as e:
    print(f"  预期报错: {type(e).__name__}: {str(e)[:60]}...")


# ============================================================
# 4. 矩阵乘法：条件判断与计算
# ============================================================
print_separator("4. 矩阵乘法（matmul / @ 运算符）")

a = torch.randn(2, 3)  # 2行3列
b = torch.randn(3, 4)  # 3行4列

print(f"矩阵 a 的 shape: {a.shape}")
print(f"矩阵 b 的 shape: {b.shape}")
print(f"\n能否相乘？判断规则：前一个的列数 == 后一个的行数")
print(f"  a 的列数 = {a.shape[1]}  vs  b 的行数 = {b.shape[0]}")
can_multiply = a.shape[1] == b.shape[0]
print(f"  结论: {'✅ 可以相乘' if can_multiply else '❌ 不能相乘'}")

# 使用 @ 运算符进行矩阵乘法
c = a @ b
print(f"\na @ b 结果 c 的 shape: {c.shape}")
print(f"  解释: {a.shape[0]}行 × {b.shape[1]}列 = ({a.shape[0]}, {b.shape[1]})")

# 用 torch.matmul 也是一样的
c2 = torch.matmul(a, b)
print(f"torch.matmul(a, b) 结果一致: {torch.allclose(c, c2)}")

# 反例：不满足条件时
d = torch.randn(2, 4)
print(f"\n反例：a.shape={a.shape}, d.shape={d.shape}")
print(
    f"  a 的列数 = {a.shape[1]}  vs  d 的行数 = {d.shape[0]} → {'✅' if a.shape[1]==d.shape[0] else '❌ 不能相乘'}"
)
try:
    bad = a @ d
except RuntimeError as e:
    print(f"  预期报错: {type(e).__name__}: {str(e)[:80]}...")


# ============================================================
# 5. 验收标准小测验（运行时打印）
# ============================================================
print_separator("5. Day 8 验收标准要点")

quiz = [
    (
        "[32, 128] 两个数字的含义",
        "→ 第一个数字 32 = batch_size（一次处理32个样本）；第二个数字 128 = 每个样本的特征维度（每个样本用128个数表示）",
    ),
    (
        "如何判断两个矩阵能否相乘",
        "→ 规则：第一个矩阵的『列数』必须等于第二个矩阵的『行数』。即 a.shape[1] == b.shape[0]",
    ),
    (
        "参数为什么具有 dtype 和 device",
        "→ dtype：决定精度（float32/float16）与显存/内存占用；device：决定参数存放位置（CPU内存 / GPU显存），数据和模型必须同设备才能计算",
    ),
    (
        "推理为什么不需要保存反向传播信息",
        "→ 推理（inference）只做前向计算（预测），不会调用 loss.backward() 更新参数。"
        "保存反向传播信息（计算图、中间激活的梯度）非常耗显存和算力，"
        "torch.inference_mode() / torch.no_grad() 会关闭梯度追踪，大幅节省显存并提速",
    ),
]

for q, a_text in quiz:
    print(f"\nQ: {q}")
    print(f"   {a_text}")

print("\n" + "=" * 60)
print("tensor_basics.py 运行完成")