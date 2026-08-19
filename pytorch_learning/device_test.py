import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl

# 思源黑体（Noto Sans SC）— Google/Adobe 开源中文字体，风格接近微软雅黑
import matplotlib.font_manager as fm

fm.fontManager.addfont("/Users/zhengwenze/.fonts/NotoSansSC-Regular.ttf")
mpl.rcParams["font.family"] = ["Noto Sans SC", "PingFang SC", "Arial Unicode MS"]
mpl.rcParams["axes.unicode_minus"] = False

x = torch.linspace(-5, 5, 100)

# --- 1. 无激活函数：纯线性变换 y = Wx + b ---
# 两层 Linear 不加激活函数，等价于一层线性变换
linear1 = nn.Linear(1, 1)
linear2 = nn.Linear(1, 1)
y_no_relu = linear2(linear1(x.unsqueeze(-1))).squeeze().detach()

# --- 2. 有激活函数：ReLU 在中间引入非线性 ---
y_relu = nn.ReLU()(x)

# --- 3. 多层 Linear + ReLU：能拟合非线性曲线 ---
layers = nn.Sequential(
    nn.Linear(1, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)
y_deep = layers(x.unsqueeze(-1)).squeeze().detach()

# --- 绘图对比 ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# 子图1：无激活函数 — 直线
axes[0].plot(x.numpy(), y_no_relu.numpy(), "b-", linewidth=2)
axes[0].grid(True)
axes[0].set_title("无激活函数：两层Linear = 一条直线\n（堆再多层还是直线）")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

# 子图2：ReLU 曲线
axes[1].plot(x.numpy(), y_relu.numpy(), "r-", linewidth=2)
axes[1].grid(True)
axes[1].set_title("ReLU：在0处折角\n（非线性的来源）")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")

# 子图3：Linear+ReLU 堆叠 — 非线性曲线
axes[2].plot(x.numpy(), y_deep.numpy(), "g-", linewidth=2)
axes[2].grid(True)
axes[2].set_title("Linear+ReLU 多层堆叠：曲线\n（能拟合复杂模式）")
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")

plt.tight_layout()
plt.show()
