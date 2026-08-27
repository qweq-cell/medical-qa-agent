from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    'bolt://44.203.170.211:7687',
    auth=('neo4j', 'interfaces-projectiles-flash')
)

with driver.session() as s:
    print("=== 节点统计 ===")
    for r in s.run('MATCH (n) RETURN labels(n)[0] as label, count(n) as cnt ORDER BY cnt DESC').data():
        print(f"  {r['label']}: {r['cnt']:,}")

    print()
    print("=== 关系统计 ===")
    for r in s.run('MATCH ()-[r]->() RETURN type(r) as t, count(r) as cnt ORDER BY cnt DESC').data():
        print(f"  {r['t']}: {r['cnt']:,}")

    total = s.run('MATCH (n) RETURN count(n) as c').single()['c']
    total_rel = s.run('MATCH ()-[r]->() RETURN count(r) as c').single()['c']
    print(f"\n总计: {total:,} 节点, {total_rel:,} 关系")

driver.close()
print("[OK]")
