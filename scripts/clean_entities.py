import pandas as pd
import re
import os



def check_dirty(name):
    reasons = []
    if name.isdigit():
        reasons.append('纯数字')
    if ':' in name or '∶' in name:
        reasons.append('含比例符号')
    if len(name) <= 2 and not re.search(r'[\u4e00-\u9fff]', name):
        reasons.append('短英文缩写')
    if not re.search(r'[\u4e00-\u9fff]', name):
        reasons.append('无中文')
    return reasons


#df['len'] = df['实体名称'].str.len()
#df_sorted = df.sort_values('len')
#print(df_sorted.head(30))  # 最短的30条 → 找垃圾
#print(df_sorted.tail(30))  # 最长的30条 → 找异常

report=[]
os.makedirs('clean', exist_ok=True)
files = ['entity_药品.csv', 'entity_疾病.csv', 'entity_症状.csv', 'entity_并发症.csv','entity_检查项目.csv','entity_科室.csv','entity_食物.csv']
for filename in files:
    df=pd.read_csv(filename)
    df['问题'] = df['实体名称'].apply(check_dirty)
    df_clean = df[df['问题'].apply(len) == 0]  # 没问题的
    df_dirty = df[df['问题'].apply(len) > 0]   # 有问题的
    report.append((filename, len(df), len(df_clean), len(df_dirty)))
    df_clean.to_csv(f'clean/{filename}', index=False)  # 存干净数据
    df_dirty.to_csv(f'clean/{filename}_removed.csv', index=False)  # 存被删的供复查

print("=" * 50)
print("数据清洗报告")
print("=" * 50)
for filename, total, clean, dirty in report:
    print(f"{filename}: {total} → {clean} (移除 {dirty} 条)")
print("=" * 50)
print(f"总计移除: {sum(r[3] for r in report)} 条")

