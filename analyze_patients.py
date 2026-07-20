"""
第一周周末检查点1：用Python读取CSV文件并做基本统计
使用内置csv模块（不依赖pandas/numpy），零额外依赖
"""
import csv
from collections import Counter

# ====== 1. 读取CSV文件 ======
rows = []
with open("patients.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # 把数字字段转成数值类型
        row['年龄'] = int(row['年龄'])
        row['住院天数'] = int(row['住院天数'])
        row['检查项目数'] = int(row['检查项目数'])
        row['药品数'] = int(row['药品数'])
        row['费用'] = float(row['费用'])
        rows.append(row)

print("=" * 50)
print("[OK] 成功读取CSV文件！")
print(f"共 {len(rows)} 条记录")
print()

# ====== 2. 查看前5条数据 ======
print("=" * 50)
print("[预览] 前5条记录：")
print(f"{'姓名':<8} {'年龄':<6} {'性别':<6} {'诊断':<14} {'住院':<6} {'费用':<10}")
print("-" * 55)
for r in rows[:5]:
    print(f"{r['姓名']:<8} {r['年龄']:<6} {r['性别']:<6} {r['诊断']:<14} {r['住院天数']:<6} {r['费用']:<10}")
print()

# ====== 3. 基本统计 ======
ages = [r['年龄'] for r in rows]
costs = [r['费用'] for r in rows]
days = [r['住院天数'] for r in rows]

print("=" * 50)
print("[统计] 患者基本统计：")
print(f"平均年龄: {sum(ages)/len(ages):.1f} 岁")
print(f"最大年龄: {max(ages)} 岁")
print(f"最小年龄: {min(ages)} 岁")
print(f"平均住院: {sum(days)/len(days):.1f} 天")
print(f"平均费用: {sum(costs)/len(costs):.2f} 元")
print(f"最高费用: {max(costs):.2f} 元")
print(f"最低费用: {min(costs):.2f} 元")
print()

# ====== 4. 按性别统计 ======
male = [r for r in rows if r['性别'] == '男']
female = [r for r in rows if r['性别'] == '女']
print(f"男性患者: {len(male)} 人, 平均年龄 {sum(r['年龄'] for r in male)/len(male):.1f} 岁")
print(f"女性患者: {len(female)} 人, 平均年龄 {sum(r['年龄'] for r in female)/len(female):.1f} 岁")
print()

# ====== 5. 手术统计 ======
surgery = [r for r in rows if r['是否手术'] == '是']
print(f"手术患者: {len(surgery)} 人, 平均费用 {sum(r['费用'] for r in surgery)/len(surgery):.2f} 元")
print(f"非手术患者: {len(rows)-len(surgery)} 人")
print()

# ====== 6. 诊断分布 ======
diagnoses = Counter(r['诊断'] for r in rows)
print("[统计] 诊断分布：")
for disease, count in diagnoses.most_common():
    print(f"  {disease}: {count} 人")
