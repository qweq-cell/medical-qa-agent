"""
第2周周三：构建医疗实体词表
从医疗知识图谱+药典数据中提取标准化的医疗实体词表
并生成BIO标注所需的词典文件
"""
import json
import csv
import os
import re
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
KG_FILE = os.path.join(DATA_DIR, "QASystemOnMedicalKG", "data", "medical.json")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ============================================================
# 1. 加载知识图谱数据
# ============================================================
print("=" * 60)
print("📂 加载医疗知识图谱...")
diseases = []
with open(KG_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            diseases.append(json.loads(line))
print(f"   {len(diseases):,} 条疾病记录")

# ============================================================
# 2. 提取并清洗各类实体
# ============================================================
print()
print("=" * 60)
print("🧹 提取并清洗实体词表")

def clean_entity_name(name):
    """清洗实体名称"""
    name = name.strip()
    name = re.sub(r'\s+', '', name)  # 去除所有空白
    name = re.sub(r'[（(][^)）]*[)）]', '', name)  # 去除括号内容
    # 去除特殊字符
    name = name.replace('"', '').replace("'", "")
    name = name.replace('℉', '度')  # 温度符号
    return name

# 提取实体
entities = {
    "疾病": set(),
    "症状": set(),
    "药品": set(),
    "检查": set(),
    "科室": set(),
    "食物": set(),
    "并发症": set(),
    "治疗方法": set(),
}

for d in diseases:
    # 疾病名
    name = clean_entity_name(d.get("name", ""))
    if name:
        entities["疾病"].add(name)

    # 症状
    for s in d.get("symptom", []):
        s = clean_entity_name(s)
        if s and len(s) >= 2 and len(s) <= 15:
            entities["症状"].add(s)

    # 药品
    for drug_list in ["common_drug", "recommand_drug"]:
        for drug in d.get(drug_list, []):
            drug = clean_entity_name(drug)
            if drug and len(drug) >= 2:
                entities["药品"].add(drug)

    # 检查项目
    for check in d.get("check", []):
        check = clean_entity_name(check)
        if check and len(check) >= 2:
            entities["检查"].add(check)

    # 科室
    for dept in d.get("cure_department", []):
        dept = clean_entity_name(dept)
        if dept and len(dept) >= 2:
            entities["科室"].add(dept)

    # 食物
    for food_list in ["do_eat", "not_eat", "recommand_eat"]:
        for food in d.get(food_list, []):
            food = clean_entity_name(food)
            if food and len(food) >= 2:
                entities["食物"].add(food)

    # 并发症
    for acc in d.get("acompany", []):
        acc = clean_entity_name(acc)
        if acc and len(acc) >= 2:
            entities["并发症"].add(acc)

    # 治疗方法
    for cure in d.get("cure_way", []):
        cure = clean_entity_name(cure)
        if cure and len(cure) >= 2:
            entities["治疗方法"].add(cure)

# 移除空字符串
for k in entities:
    entities[k].discard("")

# 统计
print()
print(f"{'实体类型':<15} {'原始数量':<10} {'说明'}")
print("-" * 60)
for etype in ["疾病", "症状", "药品", "检查", "科室", "食物", "并发症", "治疗方法"]:
    print(f"{etype:<15} {len(entities[etype]):<10,}")

total = sum(len(v) for v in entities.values())
print(f"\n{'总计':<15} {total:<10,} 个标准化实体")

# ============================================================
# 3. 实体编码表（BIO标注用）
# ============================================================
print()
print("=" * 60)
print("🏷️  生成BIO标注标签编码")

bio_labels = {}
label_id = 0

# B-开头（实体开始），I-开头（实体内部），O（非实体）
for etype in ["疾病", "症状", "药品", "检查", "科室", "食物", "并发症", "治疗方法"]:
    b_label = f"B-{etype}"
    i_label = f"I-{etype}"
    bio_labels[b_label] = label_id
    label_id += 1
    bio_labels[i_label] = label_id
    label_id += 1

bio_labels["O"] = label_id  # 非实体

print(f"{'标签':<20} {'ID':<5}")
print("-" * 30)
for label, lid in bio_labels.items():
    print(f"{label:<20} {lid:<5}")

# 保存标签映射
label_file = os.path.join(PROCESSED_DIR, "bio_labels.csv")
with open(label_file, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "id"])
    for label, lid in sorted(bio_labels.items(), key=lambda x: x[1]):
        writer.writerow([label, lid])
print(f"\n   ✅ 标签映射已保存: {label_file}")

# ============================================================
# 4. 生成综合实体词典（JSON格式）
# ============================================================
print()
print("=" * 60)
print("📖 生成综合实体词典")

entity_dict = {}
for etype, ent_set in entities.items():
    entity_dict[etype] = sorted(list(ent_set))

dict_file = os.path.join(PROCESSED_DIR, "medical_entity_dict.json")
with open(dict_file, "w", encoding="utf-8") as f:
    json.dump(entity_dict, f, ensure_ascii=False, indent=2)

print(f"   ✅ 综合实体词典: {dict_file}")

# ============================================================
# 5. 补充：根据ICD-10大类手动构建疾病分类体系
# ============================================================
print()
print("=" * 60)
print("📋 构建疾病分类体系（参考ICD-10大类）")

icd10_categories = {
    "A00-B99": "某些传染病和寄生虫病",
    "C00-D48": "肿瘤",
    "D50-D89": "血液及造血器官疾病",
    "E00-E90": "内分泌、营养和代谢疾病",
    "F00-F99": "精神和行为障碍",
    "G00-G99": "神经系统疾病",
    "H00-H59": "眼和附器疾病",
    "H60-H95": "耳和乳突疾病",
    "I00-I99": "循环系统疾病",
    "J00-J99": "呼吸系统疾病",
    "K00-K93": "消化系统疾病",
    "L00-L99": "皮肤和皮下组织疾病",
    "M00-M99": "肌肉骨骼系统和结缔组织疾病",
    "N00-N99": "泌尿生殖系统疾病",
    "O00-O99": "妊娠、分娩和产褥期",
    "P00-P96": "起源于围生期的某些情况",
    "Q00-Q99": "先天性畸形、变形和染色体异常",
    "R00-R99": "症状、体征和异常临床所见",
    "S00-T98": "损伤、中毒和外因",
    "V01-Y98": "疾病和死亡的外因",
    "Z00-Z99": "影响健康状态的因素",
    "U00-U99": "特殊目的代码",
}

# 分析疾病分类分布
category_counts = {}
for d in diseases:
    cat = d.get("category", [])
    if isinstance(cat, list):
        for c in cat:
            if c not in ["疾病百科"]:
                category_counts[c] = category_counts.get(c, 0) + 1

print(f"{'分类':<30} {'疾病数':<10}")
print("-" * 50)
for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
    bar = "█" * (count // 50)
    print(f"{cat:<30} {count:<10,} {bar}")

# 保存分类体系
cat_file = os.path.join(PROCESSED_DIR, "disease_categories.csv")
with open(cat_file, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["分类", "疾病数", "ICD10编码范围"])
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        writer.writerow([cat, count, ""])

print(f"\n   ✅ 疾病分类已保存: {cat_file}")

# ============================================================
# 6. 中文简繁体转换（统一为简体）
# ============================================================
# 医疗数据中可能存在繁体字，需要统一处理
# 这里记录一下，实际转换可用 opencc 库
print()
print("=" * 60)
print("📝 词表清洗统计")

# 统计药品名称中的剂型关键词
dosage_forms = ["片", "胶囊", "注射液", "颗粒", "口服液", "丸", "膏", "散", "糖浆",
                "滴眼液", "喷雾", "栓", "贴剂", "软膏", "乳膏", "凝胶"]

drug_with_form = 0
for drug in entities["药品"]:
    for form in dosage_forms:
        if form in drug:
            drug_with_form += 1
            break

print(f"   药品中含剂型信息: {drug_with_form}/{len(entities['药品'])} "
      f"({drug_with_form/len(entities['药品'])*100:.1f}%)")

# 检查项目分类
check_keywords = ["CT", "MRI", "B超", "X线", "检查", "测定", "试验", "镜", "图"]
check_with_keyword = 0
for check in entities["检查"]:
    for kw in check_keywords:
        if kw in check:
            check_with_keyword += 1
            break

print(f"   检查项目含关键词: {check_with_keyword}/{len(entities['检查'])} "
      f"({check_with_keyword/len(entities['检查'])*100:.1f}%)")

# ============================================================
# 7. 导出用于NLP任务的词典（BIO标注用）
# ============================================================
print()
print("=" * 60)
print("🎯 导出NER训练用实体词典")

# 格式：每行一个实体，格式为 "实体名\t实体类型"
ner_dict_file = os.path.join(PROCESSED_DIR, "ner_entity_dict.txt")
with open(ner_dict_file, "w", encoding="utf-8") as f:
    for etype in ["疾病", "症状", "药品", "检查", "科室", "并发症"]:
        for entity in sorted(entities[etype]):
            f.write(f"{entity}\t{etype}\n")

# 统计行数
with open(ner_dict_file, "r", encoding="utf-8") as f:
    line_count = sum(1 for _ in f)
print(f"   ✅ NER实体词典: {ner_dict_file}")
print(f"      共 {line_count:,} 条实体（格式：实体名\\t实体类型）")

print()
print("=" * 60)
print("🎉 医疗实体词表构建完成！")
print(f"   实体词表: data/processed/medical_entity_dict.json")
print(f"   BIO标签:  data/processed/bio_labels.csv")
print(f"   NER词典:  data/processed/ner_entity_dict.txt")
print(f"   疾病分类: data/processed/disease_categories.csv")
