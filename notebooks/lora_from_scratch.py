"""
LoRA 从零实现 —— 一行一行讲清楚
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# 第 1 步：最核心的 LoRALayer
# ═══════════════════════════════════════════════════════════════

class LoRALayer(nn.Module):
    """
    这就是 LoRA 的核心——两个小矩阵 A 和 B。

    理论：ΔW = BA
    - W₀ 是 d×d 的大矩阵
    - A 是 d×r 的小矩阵
    - B 是 r×d 的小矩阵
    - r << d，所以 A×B 参数量远小于 W₀
    """

    def __init__(self, in_dim, out_dim, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # 缩放系数，控制 LoRA 的强度

        # A 矩阵：d×r，用随机高斯初始化
        # 为什么要随机？因为一开始不希望 LoRA 影响原模型输出，随机值很小
        self.A = nn.Parameter(torch.randn(in_dim, rank) * 0.01)

        # B 矩阵：r×d，初始化为 0
        # 初始为 0 意味着 BA = 0，LoRA 一开始不改变原模型输出
        self.B = nn.Parameter(torch.zeros(rank, out_dim))

    def forward(self, x):
        """
        LoRA 的修正量 = α/r · x @ A @ B
        - x: (batch, in_dim)
        - A: (in_dim, rank)
        - B: (rank, out_dim)
        - 结果: (batch, out_dim)

        注意：这里用的是 @（矩阵乘法），不是 *（逐元素乘）
        """
        delta = x @ self.A @ self.B          # (batch, out_dim)
        return delta * self.scaling           # 乘上 α/r

    # 关于参数量的直观对比：
    # 假设 d=4096, r=8
    # W₀: 4096×4096 = 16,777,216 个参数
    # A:  4096×8    =     32,768 个参数
    # B:  8×4096    =     32,768 个参数
    # LoRA 合计: 65,536 个参数 → 只有原来的 0.39%！


# ═══════════════════════════════════════════════════════════════
# 第 2 步：把 LoRALayer 嫁接到 Linear 上
# ═══════════════════════════════════════════════════════════════

class LinearWithLoRA(nn.Module):
    """
    把 LoRA 装进一个标准 Linear 层里。

    训练时：y = W₀x + α/r·BAx
    推理时：y = (W₀ + BA)x  —— 把 BA 合并回 W₀，零额外计算
    """

    def __init__(self, linear, rank=8, alpha=16):
        super().__init__()
        # 原权重 W₀ —— 冻结，不训练
        self.linear = linear
        self.linear.weight.requires_grad = False  # 关键！冻结原权重
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        # LoRA 旁路 —— 只训练这个
        self.lora = LoRALayer(
            in_dim=linear.in_features,
            out_dim=linear.out_features,
            rank=rank,
            alpha=alpha,
        )

    def forward(self, x):
        """
        前向传播：
        1. 原权重输出：W₀x
        2. LoRA 修正：α/r·BAx
        3. 加起来
        """
        # 原权重输出（冻结的，不更新）
        original = self.linear(x)

        # LoRA 修正量（只训练这个）
        delta = self.lora(x)

        # 加起来就是 LoRA 微调后的输出
        return original + delta

    def merge(self):
        """
        推理时调用：把 BA 合并进 W₀
        合并后 LoRA 零额外计算
        """
        # W_merged = W₀ + α/r · BA
        # BA 是 (in_dim, out_dim) 的矩阵，跟 W₀ 形状一样
        delta = (self.lora.A @ self.lora.B) * self.lora.scaling

        # 合并进原权重
        self.linear.weight.data += delta.T

        # 合并后，forward 就只用 linear，不再走 lora 分支
        # 所以我们把 forward 替换掉
        self.forward = self.linear.forward

        # 清掉 LoRA 权重，释放显存（可选）
        del self.lora


# ═══════════════════════════════════════════════════════════════
# 第 3 步：验证——在 MNIST 上对比全量微调 vs LoRA
# ═══════════════════════════════════════════════════════════════

# 为了演示，我们模拟一个"预训练"模型
# 先训一个简单模型，然后看 LoRA 微调的效果

class SimpleModel(nn.Module):
    """一个简单的 2 层 MLP，模拟预训练模型"""
    def __init__(self, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ═══════════════════════════════════════════════════════════════
# 验证 LoRA 的参数冻结
# ═══════════════════════════════════════════════════════════════

def check_lora_params():
    """验证 LoRA 确实只训练了极小部分参数"""
    print("=" * 50)
    print("验证：LoRA 的参数冻结")
    print("=" * 50)

    # 创建一个 Linear + LoRA
    linear = nn.Linear(4096, 4096)  # 模拟 Transformer 的 Q 矩阵
    model = LinearWithLoRA(linear, rank=8, alpha=16)

    # 统计可训练参数
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"总参数量:      {total:,}")
    print(f"可训练参数量:  {trainable:,}")
    print(f"LoRA 占比:     {trainable/total*100:.2f}%")
    print(f"节省:          {total/trainable:.0f}x")
    print()

    # 验证合并
    print("合并前 linear.weight 第一个值:", model.linear.weight[0, 0].item())
    model.merge()
    print("合并后 linear.weight 第一个值:", model.linear.weight[0, 0].item())
    print("合并后 forward 是否还是 LoRA:", "LoRA" in str(model.forward))
    print()


if __name__ == "__main__":
    check_lora_params()