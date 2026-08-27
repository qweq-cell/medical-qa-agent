#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导入医疗知识图谱到本地 Neo4j
数据源: data/QASystemOnMedicalKG/data/medical.json (8808 条疾病记录, UTF-8 JSONL)
目标:   bolt://127.0.0.1:7687 (本地 Neo4j)

用法:
    python scripts/import_medical_kg.py [--password 12345678]

特点:
    - 全部使用 MERGE, 幂等可重复运行
    - 批量 UNWIND 导入 (远快于逐条 create)
    - 7 类节点: Disease/Symptom/Drug/Check/Food/Department/Producer
    - 11 种关系: has_symptom/acompany_with/common_drug/recommand_drug/
      need_check/no_eat/do_eat/recommand_eat/belongs_to(病-科,科-科)/drugs_of
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from neo4j import GraphDatabase

BASE_DIR = Path(__file__).resolve().parent.parent
KG_JSON = BASE_DIR / "data" / "QASystemOnMedicalKG" / "data" / "medical.json"

BATCH = 500


def load_data():
    """读取 medical.json, 返回节点集合与关系列表"""
    diseases = []            # 疾病 dict(name + 属性)
    nodes = {"Disease": set(), "Symptom": set(), "Drug": set(),
             "Check": set(), "Food": set(), "Department": set(), "Producer": set()}
    rels = {"has_symptom": [], "acompany_with": [], "common_drug": [],
            "recommand_drug": [], "need_check": [], "no_eat": [],
            "do_eat": [], "recommand_eat": [], "belongs_to": [], "drugs_of": []}

    with open(KG_JSON, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            name = d["name"]
            diseases.append(d)
            nodes["Disease"].add(name)

            for s in d.get("symptom", []):
                nodes["Symptom"].add(s)
                rels["has_symptom"].append((name, s))
            for c in d.get("acompany", []):
                nodes["Disease"].add(c)
                rels["acompany_with"].append((name, c))
            for dr in d.get("common_drug", []):
                nodes["Drug"].add(dr)
                rels["common_drug"].append((name, dr))
            for dr in d.get("recommand_drug", []):
                nodes["Drug"].add(dr)
                rels["recommand_drug"].append((name, dr))
            for ck in d.get("check", []):
                nodes["Check"].add(ck)
                rels["need_check"].append((name, ck))
            for fd in d.get("not_eat", []):
                nodes["Food"].add(fd)
                rels["no_eat"].append((name, fd))
            for fd in d.get("do_eat", []):
                nodes["Food"].add(fd)
                rels["do_eat"].append((name, fd))
            for fd in d.get("recommand_eat", []):
                nodes["Food"].add(fd)
                rels["recommand_eat"].append((name, fd))

            # 科室: 疾病->科室, 以及大小科室归属
            depts = d.get("cure_department", [])
            if depts:
                if len(depts) == 1:
                    rels["belongs_to"].append((name, depts[0]))
                elif len(depts) >= 2:
                    rels["belongs_to"].append((name, depts[-1]))
                    rels["belongs_to"].append((depts[-1], depts[0]))  # 小科->大科
                for dep in depts:
                    nodes["Department"].add(dep)

            # 厂商 -> 药品
            for dd in d.get("drug_detail", []):
                if "(" not in dd:
                    continue
                producer, drug = dd.split("(", 1)
                drug = drug.replace(")", "")
                nodes["Producer"].add(producer)
                nodes["Drug"].add(drug)
                rels["drugs_of"].append((producer, drug))

    return diseases, nodes, rels


def import_nodes(driver, nodes):
    """批量 MERGE 节点"""
    total = 0
    for label, names in nodes.items():
        if not names:
            continue
        name_list = sorted(names)
        for i in range(0, len(name_list), BATCH):
            batch = [{"name": n} for n in name_list[i:i + BATCH]]
            with driver.session() as s:
                s.run(
                    f"UNWIND $batch AS row MERGE (n:`{label}` {{name: row.name}})",
                    batch=batch,
                )
        total += len(name_list)
        print(f"  {label}: {len(name_list):,}")
    return total


def import_disease_props(driver, diseases):
    """为 Disease 节点补充属性"""
    props = ["desc", "prevent", "cause", "cure_way", "cure_lasttime",
             "cured_prob", "easy_get", "cure_department", "get_prob"]
    for i in range(0, len(diseases), BATCH):
        batch = []
        for d in diseases[i:i + BATCH]:
            row = {"name": d["name"]}
            for p in props:
                v = d.get(p)
                if isinstance(v, (list, tuple)):
                    v = " ".join(str(x) for x in v)
                row[p] = v or ""
            batch.append(row)
        with driver.session() as s:
            s.run(
                """UNWIND $batch AS row
                   MERGE (d:Disease {name: row.name})
                   SET d.desc = row.desc, d.prevent = row.prevent, d.cause = row.cause,
                       d.cure_way = row.cure_way, d.cure_lasttime = row.cure_lasttime,
                       d.cured_prob = row.cured_prob, d.easy_get = row.easy_get,
                       d.cure_department = row.cure_department, d.get_prob = row.get_prob""",
                batch=batch,
            )


def import_rels(driver, rels):
    """批量 MERGE 关系"""
    patterns = {
        "has_symptom": ("Disease", "Symptom", "has_symptom"),
        "acompany_with": ("Disease", "Disease", "acompany_with"),
        "common_drug": ("Disease", "Drug", "common_drug"),
        "recommand_drug": ("Disease", "Drug", "recommand_drug"),
        "need_check": ("Disease", "Check", "need_check"),
        "no_eat": ("Disease", "Food", "no_eat"),
        "do_eat": ("Disease", "Food", "do_eat"),
        "recommand_eat": ("Disease", "Food", "recommand_eat"),
        "drugs_of": ("Producer", "Drug", "drugs_of"),
    }
    total = 0
    for key, (src_label, dst_label, rel_type) in patterns.items():
        edges = rels[key]
        if not edges:
            continue
        # 去重
        uniq = sorted(set(edges))
        for i in range(0, len(uniq), BATCH):
            batch = [{"a": a, "b": b} for a, b in uniq[i:i + BATCH]]
            with driver.session() as s:
                s.run(
                    f"""UNWIND $batch AS row
                        MATCH (a:`{src_label}` {{name: row.a}}), (b:`{dst_label}` {{name: row.b}})
                        MERGE (a)-[r:`{rel_type}`]->(b)""",
                    batch=batch,
                )
        total += len(uniq)
        print(f"  {rel_type}: {len(uniq):,}")
    # belongs_to 单独处理(两端可能是 Disease 或 Department)
    uniq = sorted(set(rels["belongs_to"]))
    for i in range(0, len(uniq), BATCH):
        batch = [{"a": a, "b": b} for a, b in uniq[i:i + BATCH]]
        with driver.session() as s:
            s.run(
                """UNWIND $batch AS row
                   MATCH (a {name: row.a}), (b {name: row.b})
                   MERGE (a)-[r:belongs_to]->(b)""",
                batch=batch,
            )
    total += len(uniq)
    print(f"  belongs_to: {len(uniq):,}")
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="bolt://127.0.0.1:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="12345678")
    args = parser.parse_args()

    print("[1/4] 读取 medical.json ...")
    diseases, nodes, rels = load_data()
    print(f"  疾病记录: {len(diseases):,} 条")

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))

    print("[2/4] 导入节点 ...")
    n_nodes = import_nodes(driver, nodes)
    print(f"  节点合计: {n_nodes:,}")

    print("[3/4] 补充疾病属性 ...")
    import_disease_props(driver, diseases)

    print("[4/4] 导入关系 ...")
    n_rels = import_rels(driver, rels)
    print(f"  关系合计: {n_rels:,}")

    driver.close()
    print("[完成] 知识图谱导入成功!")


if __name__ == "__main__":
    main()
