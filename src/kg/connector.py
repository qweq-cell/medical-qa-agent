"""
Neo4j 知识图谱连接器
"""
from neo4j import GraphDatabase


class KGConnector:
    """Neo4j 知识图谱连接器"""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def query(self, cypher: str, params: dict = None) -> list[dict]:
        """执行 Cypher 查询"""
        with self.driver.session() as session:
            result = session.run(cypher, params or {})
            return [r.data() for r in result]