# MedGuardian 规则质控评测报告

- 数据集：`eval/cases.json`（25 例，合成埋点，见文件 meta）
- 模式：rule-only（无 Neo4j / 无 LLM）｜ 日期：2026-09-09

## 汇总

| 指标 | 值 |
|---|---|
| 植入问题实例（期望） | 21 |
| 检出 | 21 |
| **Recall（检出率）** | **100.0%** |
| 误报（非期望抛出） | 0 |
| **Precision** | **100.0%** |
| 干净对照组误报病历 | 0/7 |
| 完全正确病历（无漏检无误报） | 25/25 |
| 延迟 mean/p50/p95(ms) | 0.1/0.1/0.3 |

### 分类型检出率

| 类型 | 期望 | 命中 | Recall |
|---|---|---|---|
| dosage_abnormal | 6 | 6 | 100% |
| gender_mismatch | 8 | 8 | 100% |
| missing_info | 7 | 7 | 100% |

## 逐例明细

| id | 期望 | 实际抛出 | 命中 | 误报 | 延迟ms | 完全正确 |
|---|---|---|---|---|---|---|
| C1 clean: 男57 高血压 | — | — | — | — | 0.7 | ✅ |
| C2 clean: 女48 糖尿病 | — | — | — | — | 0.2 | ✅ |
| C3 clean: 男61 高血压 | — | — | — | — | 0.1 | ✅ |
| C4 clean: 女55 糖尿病 | — | — | — | — | 0.1 | ✅ |
| C5 clean: 男59 糖尿病+高血压 | — | — | — | — | 0.1 | ✅ |
| C6 clean: 女66 高血压 | — | — | — | — | 0.1 | ✅ |
| G1 gender: 男56 + 子宫肌瘤 | gender_mismatch | gender_mismatch | gender_mismatch | — | 0.1 | ✅ |
| G2 gender: 女63 + 前列腺增生 | gender_mismatch | gender_mismatch | gender_mismatch | — | 0.1 | ✅ |
| G3 gender: 男49 + 卵巢囊肿 | gender_mismatch | gender_mismatch | gender_mismatch | — | 0.3 | ✅ |
| G4 gender+dosage: 男 + 乳腺增生 + 二甲双胍3000mg | dosage_abnormal、gender_mismatch | dosage_abnormal、gender_mismatch | dosage_abnormal、gender_mismatch | — | 0.4 | ✅ |
| G5 gender: 男45 + 输卵管炎 | gender_mismatch | gender_mismatch | gender_mismatch | — | 0.2 | ✅ |
| G6 gender+missing: 女68 + 前列腺增生 + 无年龄 | gender_mismatch、missing_info | gender_mismatch、missing_info | gender_mismatch、missing_info | — | 0.1 | ✅ |
| G7 gender: 男39 + 阴道炎 | gender_mismatch | gender_mismatch | gender_mismatch | — | 0.3 | ✅ |
| D1 dosage: 二甲双胍5000mg | dosage_abnormal | dosage_abnormal | dosage_abnormal | — | 0.1 | ✅ |
| D2 dosage: 硝苯地平2000mg | dosage_abnormal | dosage_abnormal | dosage_abnormal | — | 0.0 | ✅ |
| D3 dosage: 二甲双胍2500mg bid | dosage_abnormal | dosage_abnormal | dosage_abnormal | — | 0.0 | ✅ |
| D4 dosage: 氨氯地平1500mg | dosage_abnormal | dosage_abnormal | dosage_abnormal | — | 0.0 | ✅ |
| D5 dosage: 阿司匹林3000mg | dosage_abnormal | dosage_abnormal | dosage_abnormal | — | 0.0 | ✅ |
| D6 dosage边界: 1000mg 应不报(>1000才报) | — | — | — | — | 0.0 | ✅ |
| M1 missing: 无年龄 | missing_info | missing_info | missing_info | — | 0.1 | ✅ |
| M2 missing: 无诊断 | missing_info | missing_info | missing_info | — | 0.2 | ✅ |
| M3 missing: 药品无剂量 | missing_info | missing_info | missing_info | — | 0.0 | ✅ |
| M4 missing: 无年龄(女糖尿病) | missing_info | missing_info | missing_info | — | 0.1 | ✅ |
| M5 missing: 无诊断+无剂量 | missing_info | missing_info | missing_info | — | 0.0 | ✅ |
| M6 missing+gender: 女 + 前列腺增生 + 无年龄 | gender_mismatch、missing_info | gender_mismatch、missing_info | gender_mismatch、missing_info | — | 0.1 | ✅ |
