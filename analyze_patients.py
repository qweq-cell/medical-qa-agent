"""
第一周周末检查点1：用Python读取CSV文件并做基本统计
使用内置csv模块（不依赖pandas/numpy），零额外依赖
"""
import csv
from collections import Counter

# ====== 1. 读取CSV文件 ======
rows = []
with open("covid_data.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # 把数字字段转成数值类型
        row['新增确诊'] = int(row['新增确诊'])
        row['新增治愈'] = int(row['新增治愈'])
        row['死亡'] = int(row['死亡'])
        rows.append(row)

print("=" * 50)
print("[OK] 成功读取CSV文件！")
print(f"共 {len(rows)} 条记录")
print()

# ====== 2. 查看前5条数据 ======
print("=" * 50)
print("[预览] 前3条记录：")
print(f"{'城市':<8} {'新增确诊':<6} {'新增治愈':<6} {'死亡':<14} ")
print("-" * 55)
for r in rows[:3]:
    print(f"{r['省份']:<8} {r['新增确诊']:<6} {r['新增治愈']:<6} {r['死亡']:<14} ")
print()

# ====== 3. 按省份统计新增确诊 ======
ages1 = [r['新增确诊'] for r in rows if r['省份'] == '北京']
ages2 = [r['新增确诊'] for r in rows if r['省份'] == '上海']
ages3 = [r['新增确诊'] for r in rows if r['省份'] == '广东']

print("=" * 50)
print("[统计] 新增确诊：")
print(f"北京新增确诊: {sum(ages1)}")
print(f"上海新增确诊: {sum(ages2)}")
print(f"广东新增确诊: {sum(ages3)}")

print()
max_p = max(("北京", sum(ages1)), ("上海", sum(ages2)), ("广东", sum(ages3)), key=lambda x: x[1])
print(f"最多的是: {max_p[0]} ({max_p[1]} 例)")


