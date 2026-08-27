"""
周日练习：Cypher 查询 + SQL 数据统计
所有查询都通过 Python + neo4j 官方驱动执行
"""
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    'bolt://44.203.170.211:7687',
    auth=('neo4j', 'interfaces-projectiles-flash')
)

def query(title, cypher, limit=5):
    """执行查询并打印结果"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    with driver.session() as s:
        results = s.run(cypher).data()
        for i, r in enumerate(results[:limit]):
            print(f"  {r}")
        if len(results) > limit:
            print(f"  ... 共 {len(results)} 条，仅显示前 {limit} 条")
    return results


# ============================================================
# Part 1: Cypher 查询练习（Neo4j）
# ============================================================
print("\n" + "="*60)
print("  Part 1: Cypher 查询练习")
print("="*60)

# 1.1 基础查询：查疾病的属性
query("1.1 查「糖尿病」的所有属性",
    "MATCH (d:Disease {name: '糖尿病'}) RETURN d.name, d.cause, d.prevent, d.cure_way")

# 1.2 一跳关系：疾病 → 症状
query("1.2 糖尿病有哪些症状？",
    "MATCH (d:Disease {name: '糖尿病'})-[r:has_symptom]->(s:Symptom) RETURN s.name LIMIT 10")

# 1.3 一跳关系：疾病 → 药品
query("1.3 糖尿病常用什么药？",
    "MATCH (d:Disease {name: '糖尿病'})-[r:common_drug]->(dr:Drug) RETURN dr.name LIMIT 10")

# 1.4 一跳关系：疾病 → 检查
query("1.4 糖尿病需要做什么检查？",
    "MATCH (d:Disease {name: '糖尿病'})-[r:need_check]->(c:Check) RETURN c.name LIMIT 10")

# 1.5 反向查询：症状 → 哪些疾病
query("1.5 发热可能是哪些疾病？",
    "MATCH (d:Disease)-[r:has_symptom]->(s:Symptom {name: '发热'}) RETURN d.name LIMIT 10")

# 1.6 反向查询：药品 → 治哪些病
query("1.6 阿司匹林能治哪些病？",
    "MATCH (d:Disease)-[r:common_drug]->(dr:Drug {name: '阿司匹林'}) RETURN d.name LIMIT 10")

# 1.7 两跳查询：疾病 → 并发症 → 并发症的症状
query("1.7 糖尿病的并发症有哪些症状？（两跳）",
    """MATCH (d:Disease {name: '糖尿病'})-[:acompany_with]->(c:Disease)-[:has_symptom]->(s:Symptom)
       RETURN c.name as 并发症, collect(s.name)[0..3] as 部分症状 LIMIT 5""")

# 1.8 自环关系：查并发症（双向）
query("1.8 糖尿病的并发症有哪些？（含反向）",
    """MATCH (d:Disease {name: '糖尿病'})-[:acompany_with]->(c:Disease)
       RETURN c.name as 并发症
       UNION
       MATCH (c:Disease)-[:acompany_with]->(d:Disease {name: '糖尿病'})
       RETURN c.name as 并发症 LIMIT 10""")

# 1.9 食物关系
query("1.9 糖尿病忌吃/宜吃什么？",
    """MATCH (d:Disease {name: '糖尿病'})-[r:no_eat]->(f:Food) RETURN '忌吃' as 类型, f.name as 食物 LIMIT 5
       UNION ALL
       MATCH (d:Disease {name: '糖尿病'})-[r:do_eat]->(f:Food) RETURN '宜吃' as 类型, f.name as 食物 LIMIT 5""")

# 1.10 聚合统计：症状最多的 Top 10 疾病
query("1.10 症状最多的 Top 10 疾病？",
    """MATCH (d:Disease)-[:has_symptom]->(s:Symptom)
       RETURN d.name, count(s) as 症状数
       ORDER BY 症状数 DESC LIMIT 10""")

# 1.11 路径查询：疾病 → 症状 ← 另一疾病（共同症状）
query("1.11 和糖尿病有共同症状的疾病？",
    """MATCH (d:Disease {name: '糖尿病'})-[:has_symptom]->(s:Symptom)<-[:has_symptom]-(other:Disease)
       WHERE other.name <> '糖尿病'
       RETURN other.name, collect(s.name)[0..3] as 共同症状, count(s) as 共同数
       ORDER BY 共同数 DESC LIMIT 10""")


# ============================================================
# Part 2: SQL/数据统计练习（pandas 对着 JSON 数据）
# ============================================================
print("\n\n" + "="*60)
print("  Part 2: 数据统计练习（pandas + medical.json）")
print("="*60)

import json
import pandas as pd
from collections import Counter

# 加载数据
DATA_FILE = "data/QASystemOnMedicalKG/data/medical.json"
diseases = []
with open(DATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            diseases.append(json.loads(line))

print(f"\n总疾病数: {len(diseases)}")

# 2.1 字段覆盖率统计
print("\n--- 2.1 字段覆盖率 ---")
fields = ['name', 'desc', 'cause', 'prevent', 'symptom', 'acompany',
          'common_drug', 'recommand_drug', 'check', 'do_eat', 'not_eat',
          'cure_department', 'cure_way', 'cure_lasttime', 'cured_prob']
for field in fields:
    count = sum(1 for d in diseases if d.get(field))
    rate = count / len(diseases) * 100
    bar = '#' * int(rate / 2)
    print(f"  {field:20s}: {count:5d}/{len(diseases)} ({rate:5.1f}%) {bar}")

# 2.2 文本长度分布
print("\n--- 2.2 文本字段长度分布（均值/P50/P90）---")
for field in ['desc', 'cause', 'prevent']:
    lengths = sorted([len(d.get(field, '')) for d in diseases if d.get(field)])
    if lengths:
        n = len(lengths)
        print(f"  {field}: 均值={sum(lengths)//n}, P50={lengths[n//2]}, P90={lengths[int(n*0.9)]}, max={lengths[-1]}")

# 2.3 科室分布
print("\n--- 2.3 科室分布 Top 10（疾病数量）---")
dept_counter = Counter()
for d in diseases:
    for dept in d.get('cure_department', []):
        dept_counter[dept] += 1
for dept, count in dept_counter.most_common(10):
    print(f"  {dept}: {count}")

# 2.4 疾病症状数分布
print("\n--- 2.4 疾病症状数分布 ---")
symptom_counts = [len(d.get('symptom', [])) for d in diseases]
print(f"  平均每种疾病: {sum(symptom_counts)/len(symptom_counts):.1f} 个症状")
print(f"  最多: {max(symptom_counts)} 个, 最少: {min(symptom_counts)} 个")
# 症状数分布 + 质量检查
bins = [0, 1, 3, 5, 10, 20, 50, 100, 1000]
for i in range(len(bins)-1):
    c = sum(1 for x in symptom_counts if bins[i] <= x < bins[i+1])
    print(f"  [{bins[i]}-{bins[i+1]}): {c} 种疾病")

# 2.5 质量检查
print("\n--- 2.5 数据质量检查 ---")
empty_name = sum(1 for d in diseases if not d.get('name'))
no_symptom = sum(1 for d in diseases if not d.get('symptom'))
print(f"  空名称: {empty_name}")
print(f"  无症状: {no_symptom}")

# 症状异常值
weird = []
for d in diseases:
    for s in d.get('symptom', []):
        if len(s) <= 1:
            weird.append(s)
print(f"  过短症状(≤1字): {len(weird)} 个, 例: {weird[:5]}")

driver.close()
print(f"\n{'='*60}")
print("  周日练习完成！")
print(f"{'='*60}")
