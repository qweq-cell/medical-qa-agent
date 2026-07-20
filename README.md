# 🏥 MedGuardian — 智能病历结构化+质控Agent

> 一个能读懂中文电子病历、自动提取关键医疗信息、并检测临床逻辑矛盾的AI Agent。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-开发中-orange.svg)]()

---

## 🎯 一句话介绍

输入一份中文病历文本 → AI自动提取疾病/症状/药品/检查项目 → 基于医学知识图谱检测逻辑矛盾 → 输出结构化质控报告。

---

## 🏗 技术架构

```
病历文本 → NER实体识别 → 实体链接 → LLM结构化 → 知识图谱验证 → 质控报告
```

| 模块 | 技术 |
|------|------|
| NER | bert-base-chinese + HuggingFace |
| 实体链接 | Sentence-BERT 向量匹配 |
| 信息抽取 | 中文医疗LLM + LoRA微调 |
| 知识图谱 | Neo4j (1万+医疗三元组) |
| 后端 | FastAPI |
| 前端 | Streamlit |

---

## 🚀 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/YOUR_USERNAME/medical-qa-agent.git
cd medical-qa-agent

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动Demo
streamlit run src/app.py
```

---

## 📁 项目结构

```
medical-qa-agent/
├── data/           # 数据集（CHIP、cEHR-Notes等）
├── models/         # 训练好的NER/LLM模型
├── src/            # 核心代码
│   ├── ner/        # NER实体识别
│   ├── llm/        # LLM信息抽取
│   ├── kg/         # 知识图谱查询
│   └── agent/      # Agent调度逻辑
├── notebooks/      # Jupyter实验笔记
├── scripts/        # 数据预处理脚本
├── PROPOSAL.md     # 项目立项文档
└── README.md
```

---

## 📅 开发进度

- [x] **第1周** (7/14-7/20)：Python基础 + 项目立项 ✅
- [ ] **第2周** (7/21-7/27)：数据集准备 + 知识图谱
- [ ] **第3周** (7/28-8/3)：医疗NER模型训练
- [ ] **第4周** (8/4-8/10)：LLM微调 + 病历结构化
- [ ] **第5周** (8/11-8/17)：Agent核心功能
- [ ] **第6周** (8/18-8/24)：工程化 + Demo
- [ ] **第7周** (8/25-8/31)：文档 + 博客 + 简历

---

## 👤 作者

医工交叉方向求职者 | 2026秋招

---

> 📅 项目开始：2026年7月14日
