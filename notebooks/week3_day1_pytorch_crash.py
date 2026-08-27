"""
Week 3 Day 1: PyTorch 速成
===========================
目标：理解 Tensor、Autograd、nn.Module、DataLoader、训练循环
预计时间：4小时，分 5 个部分

运行方式：在 medical-qa-agent 目录下执行
    python notebooks/week3_day1_pytorch_crash.py
"""

# ================================================================
# Part 1: Tensor 基础 -- PyTorch 的 "numpy"（30分钟）
# ================================================================
print("=" * 60)
print("Part 1: Tensor 基础")
print("=" * 60)

import torch
import numpy as np

# 1.1 创建 Tensor
print("\n--- 1.1 创建 Tensor ---")
a = torch.tensor([1, 2, 3])           # 从 list
b = torch.zeros(3, 4)                 # 全0矩阵
c = torch.ones(2, 3)                  # 全1矩阵
d = torch.randn(2, 3)                 # 标准正态分布随机数
e = torch.arange(0, 10, 2)            # 等差数列
f = torch.from_numpy(np.array([1.0, 2.0, 3.0]))  # numpy -> tensor

print(f"a = {a}, shape={a.shape}, dtype={a.dtype}")
print(f"b.shape = {b.shape}")  # torch.Size([3, 4])
print(f"d (randn) =\n{d}")

# 1.2 Tensor 运算
print("\n--- 1.2 Tensor 运算 ---")
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

print(f"x + y = {x + y}")             # 逐元素加
print(f"x * y = {x * y}")             # 逐元素乘（不是矩阵乘！）
print(f"x @ y = {x @ y}")             # 点积 (1*4 + 2*5 + 3*6 = 32)
print(f"x.dot(y) = {x.dot(y)}")       # 同上

# 矩阵乘法
A = torch.randn(2, 3)  # 2行3列
B = torch.randn(3, 4)  # 3行4列
C = A @ B               # 矩阵乘 -> (2, 4)
# 等价于: C = torch.matmul(A, B) 或 C = torch.mm(A, B)
print(f"A(2,3) @ B(3,4) = C{tuple(C.shape)}")

# 1.3 索引、切片、变形
print("\n--- 1.3 索引/切片/变形 ---")
t = torch.arange(12).reshape(3, 4)   # 变形 -> 3x4
print(f"t =\n{t}")
print(f"t[0, 0] = {t[0, 0]}")        # 单个元素
print(f"t[0] = {t[0]}")              # 第一行
print(f"t[:, 0] = {t[:, 0]}")        # 第一列
print(f"t[1:, 1:3] =\n{t[1:, 1:3]}") # 切片

# view vs reshape
print(f"\nt.view(-1, 6).shape = {t.view(-1, 6).shape}")  # view 共享内存
print(f"t.unsqueeze(0).shape = {t.unsqueeze(0).shape}")  # 增加维度 (1,3,4)
print(f"t.squeeze().shape = {t.unsqueeze(0).squeeze(0).shape}")  # 去掉维度

# 1.4 设备切换（CPU / GPU）
print("\n--- 1.4 设备 ---")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前设备: {device}")
t_gpu = t.to(device)  # 移到 GPU（如果有）
print(f"t_gpu.device = {t_gpu.device}")

# 1.5 关键区别: Tensor vs numpy
print("\n--- 1.5 PyTorch vs NumPy 关键区别 ---")
print("相同: 索引、切片、reshape 语法几乎一样")
print("不同: (1) Tensor 可以在 GPU 上运行")
print("      (2) Tensor 支持自动求导 (autograd)")
print("      (3) Tensor 默认 float32，numpy 默认 float64")


# ================================================================
# Part 2: Autograd -- 自动求导（40分钟）
# ================================================================
print("\n" + "=" * 60)
print("Part 2: Autograd 自动求导")
print("=" * 60)

# 2.1 requires_grad：标记需要求导的 Tensor
print("\n--- 2.1 requires_grad ---")
x = torch.tensor([2.0, 3.0], requires_grad=True)
print(f"x = {x}, requires_grad = {x.requires_grad}")

y = x[0]**2 + x[1]**3   # y = 2^2 + 3^3 = 4 + 27 = 31
print(f"y = x[0]^2 + x[1]^3 = 2^2 + 3^3 = {y.item()}")

# 2.2 backward()：反向传播计算梯度
print("\n--- 2.2 backward() ---")
y.backward()  # 计算 dy/dx = [2*x[0], 3*x[1]^2] = [4, 27]
print(f"dy/dx = {x.grad}")  # tensor([4., 27.])
print(f"验证: dy/dx[0] = 2*2 = 4 OK")
print(f"验证: dy/dx[1] = 3*3^2 = 27 OK")

# 2.3 训练中的典型用法
print("\n--- 2.3 训练中的三步走 ---")
# 假设这是一个简单的线性模型: y = w*x + b
w = torch.tensor([3.0], requires_grad=True)  # 权重
b = torch.tensor([1.0], requires_grad=True)  # 偏置

# 前向传播
x_data = torch.tensor([2.0])
y_pred = w * x_data + b        # 预测值 = 3*2 + 1 = 7
y_true = torch.tensor([5.0])   # 真实值
loss = (y_pred - y_true) ** 2  # MSE loss = (7-5)^2 = 4

# 反向传播
loss.backward()                  # 计算 dloss/dw 和 dloss/db

# 查看梯度
# loss = (w*x+b - y_true)^2 = (w*2+1-5)^2 = (2w-4)^2
# dloss/dw = 2*(2w-4)*2 = 4*(2w-4) = 4*(6-4) = 8
# dloss/db = 2*(2w-4)*1 = 2*(2*3-4) = 4
print(f"dloss/dw = {w.grad.item():.1f} (期望: 8)")
print(f"dloss/db = {b.grad.item():.1f} (期望: 4)")

# 2.4 重要注意事项
print("\n--- 2.4 注意事项 ---")
print("[!] 每次 backward() 前要 zero_grad()，否则梯度会累加")
print("[!] 只有 requires_grad=True 的叶子节点才有 .grad")
print("[!] 评估时用 torch.no_grad() 禁用梯度计算，节省内存")


# ================================================================
# Part 3: nn.Module -- 构建神经网络（1小时）
# ================================================================
print("\n" + "=" * 60)
print("Part 3: nn.Module 构建神经网络")
print("=" * 60)

# 3.1 最简单的网络：线性回归
print("\n--- 3.1 线性回归模型 ---")

class LinearRegression(torch.nn.Module):
    """单层线性模型 y = Wx + b"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # nn.Linear 就是 y = Wx + b
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression(3, 1)  # 3个特征 -> 1个输出
print(f"模型结构:\n{model}")
print(f"参数: W.shape={model.linear.weight.shape}, b.shape={model.linear.bias.shape}")

# 3.2 一个实用的分类网络（医疗NER会用到类似的）
print("\n--- 3.2 分类网络（医疗NER原型）---")

class MedicalClassifier(torch.nn.Module):
    """示例：病历文本 -> 疾病分类"""
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=10):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)       # (batch, 768) -> (batch, 256)
        x = self.relu(x)      # 激活
        x = self.dropout(x)   # 防过拟合
        x = self.fc2(x)       # (batch, 256) -> (batch, 10)
        return x

model2 = MedicalClassifier()
print(f"分类模型参数量: {sum(p.numel() for p in model2.parameters()):,}")

# 3.3 常用层一览
print("\n--- 3.3 常用层 ---")
print("nn.Linear(in, out)   -> 全连接层（最常用）")
print("nn.Conv1d/Conv2d      -> 卷积层（图像/时序）")
print("nn.LSTM/GRU           -> 循环层（文本/序列）")
print("nn.Embedding(vocab,dim)-> 词嵌入（文本必用）")
print("nn.Dropout(p)         -> Dropout 正则化")
print("nn.BatchNorm1d/2d     -> 批归一化")
print("nn.ReLU/GELU/Tanh     -> 激活函数")
print("nn.CrossEntropyLoss() -> 分类损失函数")
print("nn.MSELoss()          -> 回归损失函数")

# 3.4 前向传播演示
print("\n--- 3.4 前向传播演示 ---")
# 模拟 BERT 输出的向量 [batch=4, hidden=768]
dummy_bert_output = torch.randn(4, 768)
logits = model2(dummy_bert_output)
print(f"输入: {dummy_bert_output.shape} -> 输出 logits: {logits.shape}")
# logits 还不是概率，需要用 softmax
probs = torch.softmax(logits, dim=1)
print(f"概率分布 (前3类): {probs[0, :3]}")
print(f"预测类别: {torch.argmax(probs, dim=1)}")


# ================================================================
# Part 4: DataLoader -- 数据加载（40分钟）
# ================================================================
print("\n" + "=" * 60)
print("Part 4: DataLoader 数据加载")
print("=" * 60)

from torch.utils.data import Dataset, DataLoader

# 4.1 自定义 Dataset（你需要为医疗数据写的！）
print("\n--- 4.1 自定义 Dataset ---")

class MedicalNERDataset(Dataset):
    """
    医疗 NER 数据集
    每条数据: 文本 + BIO标注序列
    """
    def __init__(self, texts, labels, tokenizer=None, max_len=128):
        """
        texts: list[str], 病历文本列表
        labels: list[list[int]], 每个字的BIO标签
        """
        self.texts = texts
        self.labels = labels

    def __len__(self):
        """必须实现：返回数据集大小"""
        return len(self.texts)

    def __getitem__(self, idx):
        """必须实现：返回第 idx 条数据"""
        return {
            'text': self.texts[idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 模拟数据
fake_texts = [f"患者因{['头痛','发热','咳嗽'][i%3]}就诊" for i in range(100)]
fake_labels = [torch.randint(0, 5, (10,)).tolist() for _ in range(100)]

dataset = MedicalNERDataset(fake_texts, fake_labels)
print(f"数据集大小: {len(dataset)}")
print(f"第0条: text={dataset[0]['text']}, labels={dataset[0]['labels']}")

# 4.2 DataLoader：批量加载
print("\n--- 4.2 DataLoader ---")
dataloader = DataLoader(
    dataset,
    batch_size=16,     # 每批16条
    shuffle=True,      # 每个epoch随机打乱
    num_workers=0,     # Windows 下必须设为 0
    drop_last=False    # 最后一批不够16条也保留
)

# 遍历一个batch
batch = next(iter(dataloader))
print(f"batch['text'] 数量: {len(batch['text'])}")
print(f"batch['labels'].shape: {batch['labels'].shape}")  # (16, 10)

# 4.3 collate_fn：处理不等长序列（NER 必备！）
print("\n--- 4.3 collate_fn: 处理不等长序列 ---")

def collate_fn(batch):
    """自定义批处理：padding 到相同长度"""
    texts = [item['text'] for item in batch]
    labels = [item['labels'] for item in batch]

    # padding labels 到最大长度
    max_len = max(len(l) for l in labels)
    padded_labels = torch.zeros(len(labels), max_len, dtype=torch.long)
    mask = torch.zeros(len(labels), max_len, dtype=torch.bool)
    for i, lab in enumerate(labels):
        padded_labels[i, :len(lab)] = lab
        mask[i, :len(lab)] = True  # mask 标记哪些位置是真实的

    return {
        'texts': texts,
        'labels': padded_labels,
        'attention_mask': mask
    }

dataloader2 = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)
batch2 = next(iter(dataloader2))
print(f"padded labels shape: {batch2['labels'].shape}")
print(f"attention_mask shape: {batch2['attention_mask'].shape}")


# ================================================================
# Part 5: 完整训练循环（1小时）
# ================================================================
print("\n" + "=" * 60)
print("Part 5: 完整训练循环")
print("=" * 60)

# 5.1 准备玩具数据
print("\n--- 5.1 准备数据 ---")
# 造一个分类任务：10维特征 -> 3类
X = torch.randn(1000, 10)
y = torch.randint(0, 3, (1000,))

# 划分 train/val
n_train = 800
X_train, y_train = X[:n_train], y[:n_train]
X_val, y_val = X[n_train:], y[n_train:]

train_loader = DataLoader(
    list(zip(X_train, y_train)),
    batch_size=32,
    shuffle=True
)
val_loader = DataLoader(
    list(zip(X_val, y_val)),
    batch_size=64
)

# 5.2 定义模型、损失函数、优化器
print("\n--- 5.2 模型/损失/优化器 ---")
model = torch.nn.Sequential(
    torch.nn.Linear(10, 64),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(64, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 3)
)

criterion = torch.nn.CrossEntropyLoss()   # 分类损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"损失函数: {criterion}")
print(f"优化器: {optimizer}")

# 5.3 训练循环
print("\n--- 5.3 训练 ---")
EPOCHS = 20
best_acc = 0.0

for epoch in range(EPOCHS):
    # === 训练阶段 ===
    model.train()
    train_loss = 0.0

    for batch_X, batch_y in train_loader:
        # (1) 清空梯度
        optimizer.zero_grad()

        # (2) 前向传播
        outputs = model(batch_X)       # (32, 3)
        loss = criterion(outputs, batch_y)

        # (3) 反向传播
        loss.backward()

        # (4) 更新参数
        optimizer.step()

        train_loss += loss.item()

    # (5) 调整学习率
    scheduler.step()

    avg_train_loss = train_loss / len(train_loader)

    # === 验证阶段 ===
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():  # 不计算梯度！
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    val_acc = correct / total

    # 保存最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pt')

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Acc: {val_acc:.3f} | "
              f"Best Acc: {best_acc:.3f}")

# 5.4 加载最佳模型做推理
print("\n--- 5.4 推理 ---")
model.load_state_dict(torch.load('best_model.pt'))
model.eval()

# 模拟新数据
new_input = torch.randn(5, 10)
with torch.no_grad():
    predictions = model(new_input)
    predicted_classes = torch.argmax(predictions, dim=1)

print(f"输入 shape: {new_input.shape}")
print(f"预测类别: {predicted_classes}")
print(f"logits:\n{predictions}")

# 5.5 训练循环模板（面试常考！）
print("\n" + "=" * 60)
print("[*] 训练循环模板（面试重点）")
print("=" * 60)
template = """
for epoch in range(num_epochs):
    # *** 训练阶段 ***
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()       # (1) 清梯度
        outputs = model(batch_X)    # (2) 前向传播
        loss = criterion(outputs, batch_y)  # (3) 算loss
        loss.backward()             # (4) 反向传播
        optimizer.step()            # (5) 更新参数

    # *** 验证阶段 ***
    model.eval()
    with torch.no_grad():           # 不计算梯度
        for batch in val_loader:
            ...

    # *** 保存最佳 ***
    if val_acc > best_acc:
        torch.save(model.state_dict(), 'best_model.pt')
"""
print(template)

# 清理临时文件
import os
if os.path.exists('best_model.pt'):
    os.remove('best_model.pt')

print("\n[OK] Week 3 Day 1 完成！")
print("接下来: 打开 notebooks/week3_day1_pytorch_crash.py 复习代码")
print("明天 (Day 2): HuggingFace Trainer API + Tokenizer")
