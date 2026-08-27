"""
知识图谱查询函数封装
"""
from src.kg.connector import KGConnector


class KGQueries:
    """知识图谱查询接口"""

    def __init__(self, connector: KGConnector):
        self.conn = connector

    # ---- 疾病相关 ----

    def get_disease_info(self, disease_name: str) -> dict | None:
        """获取疾病基本信息"""
        results = self.conn.query(
            "MATCH (d:Disease {name: $name}) RETURN d.name, d.cause, d.prevent, d.cure_way",
            {"name": disease_name}
        )
        return results[0] if results else None

    def get_symptoms(self, disease_name: str) -> list[str]:
        """获取疾病的症状列表"""
        results = self.conn.query(
            "MATCH (d:Disease {name: $name})-[r:has_symptom]->(s:Symptom) RETURN s.name",
            {"name": disease_name}
        )
        return [r["s.name"] for r in results]

    def get_common_drugs(self, disease_name: str) -> list[str]:
        """获取疾病常用药品"""
        results = self.conn.query(
            "MATCH (d:Disease {name: $name})-[r:common_drug]->(dr:Drug) RETURN dr.name",
            {"name": disease_name}
        )
        return [r["dr.name"] for r in results]

    def get_need_checks(self, disease_name: str) -> list[str]:
        """获取疾病需要做的检查"""
        results = self.conn.query(
            "MATCH (d:Disease {name: $name})-[r:need_check]->(c:Check) RETURN c.name",
            {"name": disease_name}
        )
        return [r["c.name"] for r in results]

    def get_complications(self, disease_name: str) -> list[str]:
        """获取并发症"""
        results = self.conn.query(
            """MATCH (d:Disease {name: $name})-[:acompany_with]->(c:Disease)
               RETURN c.name
               UNION
               MATCH (c:Disease)-[:acompany_with]->(d:Disease {name: $name})
               RETURN c.name""",
            {"name": disease_name}
        )
        return [r["c.name"] for r in results]

    # ---- 药品相关 ----

    def get_drug_diseases(self, drug_name: str) -> list[dict]:
        """获取药品能治疗的疾病（含关系类型）"""
        results = self.conn.query(
            """MATCH (d:Disease)-[r]->(dr:Drug {name: $name})
               WHERE type(r) IN ['common_drug', 'recommand_drug']
               RETURN d.name as disease, type(r) as rel_type""",
            {"name": drug_name}
        )
        return results

    # ---- 检查相关 ----

    def get_diseases_by_symptom(self, symptom_name: str) -> list[str]:
        """根据症状查询可能的疾病"""
        results = self.conn.query(
            "MATCH (d:Disease)-[:has_symptom]->(s:Symptom {name: $name}) RETURN d.name",
            {"name": symptom_name}
        )
        return [r["d.name"] for r in results]

    # ---- 统计 ----

    def get_entity_count(self) -> dict:
        """获取各实体数量统计"""
        node_results = self.conn.query(
            "MATCH (n) RETURN labels(n)[0] as label, count(n) as cnt ORDER BY cnt DESC"
        )
        rel_results = self.conn.query(
            "MATCH ()-[r]->() RETURN type(r) as t, count(r) as cnt ORDER BY cnt DESC"
        )
        return {
            "nodes": {r["label"]: r["cnt"] for r in node_results},
            "relations": {r["t"]: r["cnt"] for r in rel_results},
        }