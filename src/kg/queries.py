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

    def drug_exists(self, drug_name: str) -> bool:
        """图谱中是否存在该药品节点。

        为什么需要它：get_drug_diseases() 在「药不在图谱里」和「药在图谱里但没记录
        适应症」两种情况下都返回空列表。对质控场景来说两者语义完全不同 ——
        前者是「无法判断」，只有后者才可能是「用药与诊断不符」。
        不区分就会给正确处方报假警（例如图谱只收商品名「盐酸二甲双胍片」，
        通用名「二甲双胍」查不到，却被判成「与诊断不符」）。
        """
        results = self.conn.query(
            "MATCH (dr:Drug {name: $name}) RETURN count(dr) as cnt",
            {"name": drug_name}
        )
        return bool(results and results[0].get("cnt", 0) > 0)

    def find_drug_names(self, keyword: str, limit: int = 20) -> list[str]:
        """按关键词模糊匹配药品名 —— 方案 2 的名称回退。

        为什么需要：图谱里的药品名多为商品名/剂型名（盐酸二甲双胍片、二甲双胍格列本脲片(Ⅰ)…），
        而实体词典给的是通用名（二甲双胍），精确匹配必然落空。
        双向包含（节点名含关键词 / 关键词含节点名）+ 按名字长度降序 —— 名字越长匹配越具体。
        """
        if not keyword:
            return []
        results = self.conn.query(
            """MATCH (dr:Drug)
               WHERE dr.name CONTAINS $kw OR $kw CONTAINS dr.name
               RETURN dr.name AS name""",
            {"kw": keyword}
        )
        names = [r["name"] for r in results if r.get("name")]
        return sorted(set(names), key=len, reverse=True)[:limit]

    # ---- 检查相关 ----

    def get_diseases_by_symptom(self, symptom_name: str) -> list[str]:
        """根据症状查询可能的疾病"""
        results = self.conn.query(
            "MATCH (d:Disease)-[:has_symptom]->(s:Symptom {name: $name}) RETURN d.name",
            {"name": symptom_name}
        )
        return [r["d.name"] for r in results]

    def symptom_exists(self, symptom_name: str) -> bool:
        """图谱中是否存在该症状节点（用于区分「未收录」与「收录但无关联疾病」）"""
        results = self.conn.query(
            "MATCH (s:Symptom {name: $name}) RETURN count(s) as cnt",
            {"name": symptom_name}
        )
        return bool(results and results[0].get("cnt", 0) > 0)

    def find_symptom_names(self, keyword: str, limit: int = 20) -> list[str]:
        """按关键词模糊匹配症状名 —— 方案 2 的名称回退。

        为什么需要：实体词典用词与图谱用词经常不一致
        （词典「口渴多饮」 vs 图谱「烦渴多饮 / 口渴 / 经常口渴」），精确匹配会 0 命中。
        双向包含 + 按名字长度降序；并过滤掉长度 < 2 的节点名，避免「渴」这类短词引入噪声。
        """
        if not keyword:
            return []
        results = self.conn.query(
            """MATCH (s:Symptom)
               WHERE s.name CONTAINS $kw OR $kw CONTAINS s.name
               RETURN s.name AS name""",
            {"kw": keyword}
        )
        names = [r["name"] for r in results if r.get("name")]
        names = [n for n in names if len(n) >= 2]
        return sorted(set(names), key=len, reverse=True)[:limit]

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