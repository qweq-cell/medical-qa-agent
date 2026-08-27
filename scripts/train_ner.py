#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医疗 NER 训练（GPU/云端版）— bert-base-chinese 微调，CMeEE-V2 数据集

数据: CMeEE-V2（CBLUE 任务2，9 类医疗实体，训练集约 15000 条）
输出: models/ner-bert/（模型 + tokenizer + label 映射）

云端用法:
    pip install torch transformers
    # 数据: 上传 data/CBLUE/CMeEE-V2/CMeEE-V2/{CMeEE-V2_train,CMeEE-V2_dev}.json
    # 或从 CBLUE 官方仓库下载
    python train_ner.py --pretrained google-bert/bert-base-chinese --epochs 5 --batch 32

本地用法(无 GPU / 绕 torch 版本限制):
    python train_ner.py --pretrained models/bert-base-chinese --train_subset 4000 --epochs 2 --batch 8
"""
import argparse
import json
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
os.environ["HF_HUB_OFFLINE"] = "1" if os.path.isdir(os.path.join(BASE_DIR, "models", "bert-base-chinese")) else "0"
os.environ["TRANSFORMERS_OFFLINE"] = os.environ["HF_HUB_OFFLINE"]

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertForTokenClassification
from torch.optim import AdamW

ENTITY_TYPES = ["bod", "dis", "sym", "ite", "dru", "pro", "mic", "dep", "equ"]
LABELS = ["O"] + [f"B-{t}" for t in ENTITY_TYPES] + [f"I-{t}" for t in ENTITY_TYPES]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


def load_data(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def to_bio_labels(text, entities):
    """char 级 BIO 标签; 嵌套冲突保留较长实体"""
    n = len(text)
    labels = ["O"] * n
    ents = sorted(entities, key=lambda e: (e["start_idx"], -(e["end_idx"] - e["start_idx"])))
    used = [False] * n
    for e in ents:
        s, en = e["start_idx"], e["end_idx"]
        if en > n:
            continue
        if any(used[s:en]):
            continue
        t = e["type"]
        labels[s] = f"B-{t}"
        for i in range(s + 1, en):
            labels[i] = f"I-{t}"
        for i in range(s, en):
            used[i] = True
    return labels


class NERDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.samples = []
        for d in data:
            text = d["text"]
            char_labels = to_bio_labels(text, d.get("entities", []))
            enc = tokenizer(
                text, max_length=max_len, truncation=True,
                padding="max_length", return_offsets_mapping=True,
            )
            token_labels = []
            for s, e in enc["offset_mapping"]:
                if s == 0 and e == 0:
                    token_labels.append(-100)
                else:
                    seg = char_labels[s:e] if e <= len(char_labels) else ["O"]
                    if not seg:
                        seg = ["O"]
                    token_labels.append(LABEL2ID[max(seg, key=seg.count)])
            self.samples.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": token_labels,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return (
            torch.tensor(s["input_ids"]),
            torch.tensor(s["attention_mask"]),
            torch.tensor(s["labels"], dtype=torch.long),
        )


def evaluate(model, loader, tokenizer, id2label, device):
    """实体级 P/R/F1（已屏蔽 padding/[CLS]/[SEP] 位置的预测）"""
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for input_ids, attn, labels in loader:
            out = model(input_ids=input_ids.to(device), attention_mask=attn.to(device))
            preds = out.logits.argmax(-1).cpu()
            for b in range(input_ids.size(0)):
                gold_labels = [id2label[p] if p != -100 else "O" for p in labels[b].tolist()]
                pred_labels = []
                for pi, p in enumerate(preds[b].tolist()):
                    if labels[b][pi] == -100:
                        pred_labels.append("O")
                    else:
                        pred_labels.append(id2label[p])

                def spans(lbs):
                    sp = set()
                    cur = None
                    for i, lb in enumerate(lbs):
                        if lb.startswith("B-"):
                            if cur:
                                sp.add(cur)
                            cur = (lb[2:], i)
                        elif lb.startswith("I-") and cur and cur[0] == lb[2:]:
                            cur = (cur[0], cur[1], i) if len(cur) == 2 else (cur[0], cur[1], i)
                        else:
                            if cur:
                                sp.add(cur)
                            cur = None
                    if cur:
                        sp.add(cur)
                    return sp

                gs, ps = spans(gold_labels), spans(pred_labels)
                tp += len(gs & ps)
                fp += len(ps - gs)
                fn += len(gs - ps)
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    f1 = 2 * p * r / (p + r + 1e-9)
    return p, r, f1


def build_model(pretrained, num_labels, device):
    """加载预训练模型（本地目录绕 torch 版本限制；HF 模型名直接 from_pretrained）"""
    if os.path.isdir(pretrained):
        config = BertConfig.from_pretrained(pretrained, num_labels=num_labels)
        model = BertForTokenClassification(config)
        sd = torch.load(os.path.join(pretrained, "pytorch_model.bin"), map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=False)  # 分类头随机, backbone 预训练
        print(f"[模型] 本地加载: {pretrained} (分类头随机初始化)")
    else:
        config = BertConfig.from_pretrained(pretrained, num_labels=num_labels)
        model = BertForTokenClassification.from_pretrained(pretrained, config=config)
        print(f"[模型] HF 在线加载: {pretrained}")
    return model.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(BASE_DIR, "data", "CBLUE", "CMeEE-V2", "CMeEE-V2"))
    ap.add_argument("--pretrained", default=os.path.join(BASE_DIR, "models", "bert-base-chinese"))
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "models", "ner-bert"))
    ap.add_argument("--train_subset", type=int, default=0, help="0=全量(15000条), >0=取前N条(CPU调试用)")
    ap.add_argument("--val_subset", type=int, default=1000, help="验证集条数")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--device", default="auto", help="auto/cuda/cpu")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[设备] {device} | CUDA: {torch.cuda.is_available()}")

    print("[数据] 加载 CMeEE-V2 ...")
    train_all = load_data(os.path.join(args.data_dir, "CMeEE-V2_train.json"))
    dev_all = load_data(os.path.join(args.data_dir, "CMeEE-V2_dev.json"))
    train_data = train_all if args.train_subset <= 0 else train_all[: args.train_subset]
    dev_data = dev_all[: args.val_subset]
    print(f"  训练 {len(train_data)} 条 / 验证 {len(dev_data)} 条")

    tokenizer = BertTokenizer.from_pretrained(args.pretrained)
    train_ds = NERDataset(train_data, tokenizer, args.max_len)
    dev_ds = NERDataset(dev_data, tokenizer, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch)

    model = build_model(args.pretrained, len(LABELS), device)
    print(f"  参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M | 标签: {len(LABELS)}")

    optimizer = AdamW(model.parameters(), lr=args.lr)
    id2label = {i: l for l, i in LABEL2ID.items()}
    best_f1 = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss, steps, t0 = 0.0, 0, time.time()
        for input_ids, attn, labels in train_loader:
            optimizer.zero_grad()
            out = model(
                input_ids=input_ids.to(device),
                attention_mask=attn.to(device),
                labels=labels.to(device),
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += out.loss.item()
            steps += 1
            if steps % 100 == 0:
                el = (time.time() - t0) / steps
                print(f"  epoch{epoch + 1} step{steps}/{len(train_loader)} loss={total_loss / steps:.4f} ({el:.2f}s/step)", flush=True)
        p, r, f1 = evaluate(model, dev_loader, tokenizer, id2label, device)
        print(f"[评估] epoch{epoch + 1}: P={p:.4f} R={r:.4f} F1={f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            os.makedirs(args.out, exist_ok=True)
            model.save_pretrained(args.out, safe_serialization=True)
            tokenizer.save_pretrained(args.out)
            with open(os.path.join(args.out, "label_map.json"), "w", encoding="utf-8") as f:
                json.dump({"labels": LABELS, "label2id": LABEL2ID}, f, ensure_ascii=False, indent=2)
            print(f"  [保存] 最佳模型 -> {args.out} (F1={f1:.4f})", flush=True)

    print(f"\n[完成] 最佳 F1 = {best_f1:.4f}  →  {args.out}/")


if __name__ == "__main__":
    main()
