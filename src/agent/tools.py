"""
Agent 工具定义
"""
from src.ner.dict_matcher import DictMatcher
from src.ner.entity_linking import EntityLinker
from src.kg.queries import KGQueries
from src.qc.rules import QCRules
from src.models.schemas import StructuredRecord, MedicalEntity


class AgentTools:
    """Agent 可用的工具集合"""

    def __init__(self, matcher: DictMatcher, linker: EntityLinker,
                 kg: KGQueries = None, rules: QCRules = None):
        self.matcher = matcher
        self.linker = linker
        self.kg = kg
        self.rules = rules or QCRules()

    def extract_entities(self, text: str) -> StructuredRecord:
        """
        工具1：实体识别 + 链接
        从病历文本中提取所有医疗实体并标准化
        """
        raw_entities = self.matcher.extract(text)
        record = StructuredRecord()

        category_map = {
            "疾病": "diseases",
            "症状": "symptoms",
            "药品": "drugs",
            "检查": "tests",
            "科室": "departments",
        }

        for raw in raw_entities:
            cat = raw["category"]
            field = category_map.get(cat)
            if not field:
                continue

            std_name = self.linker.link(raw["name"], cat)
            entity = MedicalEntity(
                name=raw["name"],
                std_name=std_name,
                category=cat,
                start=raw["start"],
                end=raw["end"],
            )
            getattr(record, field).append(entity)

        return record

    def query_knowledge_graph(self, entities: StructuredRecord) -> dict:
        """
        工具2：查询知识图谱
        为识别到的实体补充 KG 信息
        """
        if not self.kg:
            return {}

        results = {}
        for disease in entities.diseases:
            name = disease.std_name
            results[name] = {
                "symptoms": self.kg.get_symptoms(name),
                "drugs": self.kg.get_common_drugs(name),
                "checks": self.kg.get_need_checks(name),
                "complications": self.kg.get_complications(name),
            }

        for drug in entities.drugs:
            name = drug.std_name
            results[name] = {
                "treats": self.kg.get_drug_diseases(name),
            }

        return results

    def run_qc_rules(self, text: str, entities: StructuredRecord) -> list:
        """
        工具3：运行质控规则
        """
        return self.rules.run_all(text, entities, self.kg)