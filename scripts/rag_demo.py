#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Demo（向量检索版）：文档 → 切分 → embedding → 向量检索 → LLM 生成

用法:
    python scripts/rag_demo.py [--query "问题"]

依赖: transformers(已装) + 本地模型 models/bge-small-zh-v1.5 + DeepSeek API(.env)
零额外依赖: 向量用 numpy + 余弦相似度（不装 faiss）
"""
import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from src.llm.client import LLMClient

MODEL_DIR = os.path.join(BASE_DIR, "models", "bge-small-zh-v1.5")

# 示例知识文档（医疗知识，模拟企业知识库）
DEMO_DOC = """2型糖尿病是一种以高血糖为特征的慢性代谢性疾病，主要由于胰岛素抵抗和胰岛素分泌不足引起。
2型糖尿病的典型症状包括多饮、多尿、多食和体重下降，即"三多一少"。
2型糖尿病的诊断标准：空腹血糖≥7.0mmol/L，或餐后2小时血糖≥11.1mmol/L，或糖化血红蛋白≥6.5%。
2型糖尿病的一线治疗药物是二甲双胍，通过减少肝糖输出、改善胰岛素敏感性来降血糖。
糖尿病治疗强调生活方式干预：控制饮食、规律运动、戒烟限酒，配合药物治疗。
高血压病是以体循环动脉压升高为主要表现的慢性病，诊断标准为收缩压≥140mmHg或舒张压≥90mmHg。
高血压治疗常用药物包括氨氯地平、硝苯地平等钙通道阻滞剂，以及卡托普利、贝那普利等ACEI类。
阿司匹林用于抗血小板治疗，可降低心脑血管事件风险，但冠心病患者使用非甾体抗炎药需谨慎。
空腹血糖反映基础胰岛素分泌水平，糖化血红蛋白反映近2-3个月平均血糖水平。"""


def embed_texts(tokenizer, model, texts, max_len=256):
    """文本 → 归一化向量（mean pooling）"""
    vecs = []
    model.eval()
    with torch.no_grad():
        for t in texts:
            enc = tokenizer(t, max_length=max_len, truncation=True, padding=True, return_tensors="pt")
            out = model(**enc)
            # mean pooling + 归一化
            mask = enc["attention_mask"].unsqueeze(-1).float()
            v = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            v = v / v.norm(dim=1, keepdim=True)
            vecs.append(v[0].numpy())
    return np.array(vecs)


def chunk_doc(doc, size=60, overlap=10):
    """按固定窗口切 chunk（含重叠，保证上下文连续）"""
    chars = list(doc.replace("\n", ""))
    chunks = []
    i = 0
    while i < len(chars):
        chunks.append("".join(chars[i:i + size]))
        i += size - overlap
    return [c for c in chunks if len(c) > 20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="2型糖尿病的症状和常用药是什么？")
    args = ap.parse_args()

    print("[1/4] 加载 embedding 模型 (bge-small-zh-v1.5) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModel.from_pretrained(MODEL_DIR)

    print("[2/4] 文档切分 + 向量化 ...")
    chunks = chunk_doc(DEMO_DOC)
    chunk_vecs = embed_texts(tokenizer, model, chunks)
    print(f"  {len(chunks)} 个 chunk, 向量维度 {chunk_vecs.shape[1]}")

    print("[3/4] 检索 Top-K ...")
    q_vec = embed_texts(tokenizer, model, [args.query])[0]
    sims = chunk_vecs @ q_vec  # 已归一化 → 余弦相似度
    top_k = 3
    idx = np.argsort(-sims)[:top_k]
    print("  命中片段:")
    for i, score in zip(idx, sims[idx]):
        print(f"    [相似度 {score:.4f}] {chunks[i][:50]}...")

    context = "\n".join(chunks[i] for i in idx)

    print("[4/4] LLM 基于检索结果生成 ...")
    llm = LLMClient()
    if not llm.configured:
        print("  (未配置 LLM_API_KEY，跳过生成；检索部分已完成)")
        return
    prompt = f"""你是知识库问答助手。请严格基于以下检索到的资料回答问题，不要编造；资料不足时说明"资料中未提及"。

【检索资料】
{context}

【问题】{args.query}

【回答】"""
    answer = llm.chat([
        {"role": "system", "content": "你是严谨的知识库问答助手，只依据给定资料回答。"},
        {"role": "user", "content": prompt},
    ], temperature=0.2, max_tokens=500)
    print("\n===== RAG 回答 =====")
    print(answer)


if __name__ == "__main__":
    main()
