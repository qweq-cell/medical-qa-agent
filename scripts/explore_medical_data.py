"""
第2周 Day2：医疗知识图谱数据探索 + 清洗
===========================================
目的：搞清楚手头的数据长什么样，为后续实体识别和知识图谱做准备。

数据来源：QASystemOnMedicalKG/data/medical.json（刘焕勇医疗KG项目）
数据格式：每行一个 JSON 对象，代表一种疾病的百科信息

探索内容：
  1. 加载数据 → 看一条完整记录长什么样
  2. 字段统计 → 哪些字段覆盖率高、哪些有缺失
  3. 实体分布 → 疾病/症状/药品/检查 各有多少个
  4. 文本长度 → desc/cause/prevent 的字符数分布
  5. 三元组统计 → 能从数据中推导出多少条知识图谱三元组
  6. 数据质量 → 空名称、无关系数据的异常记录数
  7. 导出实体词表 → 每种实体类型导出为 CSV

"""

import json
import csv
import os
import sys
from collections import Counter, defaultdict

# 修复 Windows 下 emoji 打印乱码
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ============================================================
# 0. 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # medical-qa-agent/
DATA_DIR = os.path.join(BASE_DIR, "data")
KG_FILE = os.path.join(DATA_DIR, "QASystemOnMedicalKG", "data", "medical.json")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 60)
print("📂 1. 加载数据")
print("=" * 60)

diseases = []
with open(KG_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            diseases.append(json.loads(line))

print(f"   ✅ 成功加载 {len(diseases):,} 条疾病记录\n")

# ============================================================
# 2. 看一条完整记录（了解数据结构）
# ============================================================
print("=" * 60)
print("🔍 2. 数据结构预览（第 1 条记录）")
print("=" * 60)

first = diseases[0]
for key, val in first.items():
    display_val = val
    if isinstance(val, str) and len(val) > 80:
        display_val = val[:80] + "..."
    elif isinstance(val, list):
        display_val = str(val[:5])  # 列表只显示前5个
    print(f"   {key}: {display_val}")

print(f"\n   📝 共 {len(first)} 个字段，详情见下方统计\n")

# ============================================================
# 3. 字段覆盖率统计
# ============================================================
print("=" * 60)
print("📊 3. 字段覆盖率统计")
print("=" * 60)

field_counter = Counter()
non_empty_counter = defaultdict(int)
total = len(diseases)

for d in diseases:
    for key in d:
        field_counter[key] += 1
        val = d[key]
        # 判断是否"有值"：非空字符串、非空列表
        if val and val != [] and val != "" and val != "否":
            non_empty_counter[key] += 1

print(f"   {'字段名':<22s} {'记录数':>7s} {'有值率':>8s}")
print(f"   {'-'*37}")
for field, count in field_counter.most_common():
    rate = non_empty_counter[field] / count * 100 if count > 0 else 0
    print(f"   {field:<22s} {count:>7,} {rate:>7.1f}%")

print(f"\n   ⚠️ 注意：do_eat/not_eat/recommand_eat 只在 ~5,500 条中出现（部分疾病无饮食数据）\n")

# ============================================================
# 4. 实体类型分布
# ============================================================
print("=" * 60)
print("🏥 4. 实体类型分布")
print("=" * 60)

# 定义：哪些字段里有列表型实体
LIST_FIELD_MAP = {
    "symptom":         "症状",
    "acompany":        "并发症",
    "common_drug":     "常用药品",
    "recommand_drug":  "推荐药品",
    "check":           "检查项目",
    "cure_department": "科室",
    "do_eat":          "宜吃食物",
    "not_eat":         "忌吃食物",
    "recommand_eat":   "推荐食谱",
    "cure_way":        "治疗方法",
}

entities = {label: set() for label in LIST_FIELD_MAP.values()}
entities["疾病"] = set()

for d in diseases:
    # 疾病名（字符串字段）
    name = d.get("name", "").strip()
    if name:
        entities["疾病"].add(name)

    # 列表型字段
    for field, label in LIST_FIELD_MAP.items():
        vals = d.get(field, [])
        if isinstance(vals, str):
            vals = [vals]  # 兼容字符串
        for v in vals:
            v = v.strip()
            if v:
                entities[label].add(v)

# 打印分布
print(f"   {'实体类型':<10s} {'数量':>8s}   {'示例（前3个）'}")
print(f"   {'-'*50}")
for etype in ["疾病", "症状", "并发症", "常用药品", "推荐药品",
              "检查项目", "科室", "宜吃食物", "忌吃食物", "推荐食谱", "治疗方法"]:
    eset = entities[etype]
    samples = list(eset)[:3]
    print(f"   {etype:<10s} {len(eset):>8,}   {samples}")

total_entities = sum(len(v) for v in entities.values())
print(f"\n   📌 总计 {total_entities:,} 个标准化实体（去重后）\n")

# ============================================================
# 5. 文本字段长度分布
# ============================================================
print("=" * 60)
print("📏 5. 文本字段长度分布")
print("=" * 60)

TEXT_FIELDS = ["desc", "cause", "prevent", "cure_way"]
print(f"   {'字段':<12s} {'平均':>7s} {'中位数':>7s} {'最小值':>7s} {'最大值':>7s} {'P90':>7s}")
print(f"   {'-'*47}")

for field in TEXT_FIELDS:
    lengths = [len(d.get(field, "")) for d in diseases if d.get(field, "")]
    if lengths:
        avg = sum(lengths) / len(lengths)
        sorted_lens = sorted(lengths)
        p50 = sorted_lens[len(sorted_lens) // 2]
        p90 = sorted_lens[int(len(sorted_lens) * 0.9)]
        print(f"   {field:<12s} {avg:>6.0f}字 {p50:>6}字 {min(lengths):>6}字 {max(lengths):>6}字 {p90:>6}字")

print()

# ============================================================
# 6. 可推导的三元组统计（知识图谱素材）
# ============================================================
print("=" * 60)
print("🔗 6. 可推导的知识图谱三元组")
print("=" * 60)

TRIPLE_MAP = {
    "symptom":         ("疾病", "has_symptom",         "症状"),
    "acompany":        ("疾病", "accompanies",         "疾病"),
    "common_drug":     ("疾病", "has_common_drug",     "药品"),
    "recommand_drug":  ("疾病", "has_recommend_drug",  "药品"),
    "check":           ("疾病", "needs_check",         "检查"),
    "cure_department": ("疾病", "belongs_to_dept",     "科室"),
    "do_eat":          ("疾病", "suitable_food",       "食物"),
    "not_eat":         ("疾病", "avoid_food",          "食物"),
}

triple_counts = {}
for d in diseases:
    for field, (head_type, rel_name, tail_type) in TRIPLE_MAP.items():
        vals = d.get(field, [])
        if isinstance(vals, str):
            vals = [vals] if vals else []
        triple_counts[rel_name] = triple_counts.get(rel_name, 0) + len(vals)

print(f"   {'关系名':<25s} {'数量':>8s}   {'三元组结构'}")
print(f"   {'-'*55}")
for rel_name, count in sorted(triple_counts.items(), key=lambda x: x[1], reverse=True):
    # 找到对应的头尾类型
    for field, (head_type, rn, tail_type) in TRIPLE_MAP.items():
        if rn == rel_name:
            print(f"   {rel_name:<25s} {count:>8,}   {head_type} → {rel_name} → {tail_type}")
            break

total_triples = sum(triple_counts.values())
print(f"\n   📌 可推导约 {total_triples:,} 条三元组（第6天导入 Neo4j 的数据源）\n")

# ============================================================
# 7. 数据质量检查
# ============================================================
print("=" * 60)
print("⚠️  7. 数据质量检查")
print("=" * 60)

# 空名称
empty_names = sum(1 for d in diseases if not d.get("name", "").strip())
print(f"   空疾病名称: {empty_names} 条")

# 无任何关系数据的记录
no_rel = 0
for d in diseases:
    has_data = False
    for field in TRIPLE_MAP:
        vals = d.get(field, [])
        if isinstance(vals, list) and len(vals) > 0:
            has_data = True
            break
        elif isinstance(vals, str) and vals.strip():
            has_data = True
            break
    if not has_data:
        no_rel += 1
print(f"   无任何关系数据: {no_rel} 条")

# 可疑症状（1个字或超过20个字）
all_symptoms = []
for d in diseases:
    all_symptoms.extend(d.get("symptom", []))
weird = [s for s in all_symptoms if len(s) == 1 or len(s) > 20]
if weird:
    print(f"   可疑症状名（≤1字或>20字）: {len(weird)} 个 → 示例: {weird[:5]}")

print()

# ============================================================
# 8. 导出实体词表 CSV
# ============================================================
print("=" * 60)
print("💾 8. 导出实体词表到 data/processed/")
print("=" * 60)

for etype, eset in entities.items():
    # 文件名用英文
    safe_name = etype.replace("/", "_")
    out_file = os.path.join(PROCESSED_DIR, f"entity_{safe_name}.csv")
    with open(out_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["实体名称", "实体类型"])
        for entity in sorted(eset):
            writer.writerow([entity, etype])
    print(f"   ✅ {etype}: {len(eset):,} 条 → entity_{safe_name}.csv")

print(f"\n{'=' * 60}")
print("🎉 数据探索完成！")
print(f"{'=' * 60}")
print(f"""
   数据概况:
   ├── 疾病记录: {len(diseases):,} 条
   ├── 实体总数: {total_entities:,} 个（11种类型）
   ├── 可推导三元组: {total_triples:,} 条
   └── 数据质量: 空名称 {empty_names} 条，无关系数据 {no_rel} 条
"""
)
