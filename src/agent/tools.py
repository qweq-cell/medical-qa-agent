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

    # ==================================================================
    # 以下三个是「给 LLM 调用」的薄封装。
    #
    # 为什么需要单独一层：上面三个方法的入参是 StructuredRecord（Python 对象），
    # LLM 无法产生这种参数 —— 它只能产出文本。而 Function Calling / MCP 的工具
    # 必须是「字符串进、结构化 JSON 出」的无状态形态，否则模型根本调不动。
    #
    # 这一层不做任何业务逻辑，只负责：调原方法 → 把结果转成 JSON 可序列化的 dict。
    # ==================================================================

    def tool_extract_entities(self, text: str) -> dict:
        """工具：实体识别（LLM 可调用版，入参为病历文本）"""
        rec = self.extract_entities(text)

        def dump(items) -> list[dict]:
            return [
                {"name": e.name, "standardized": e.std_name, "category": e.category}
                for e in items
            ]

        return {
            "diseases": dump(rec.diseases),
            "symptoms": dump(rec.symptoms),
            "drugs": dump(rec.drugs),
            "tests": dump(rec.tests),
            "departments": dump(rec.departments),
            "counts": {
                "diseases": len(rec.diseases),
                "symptoms": len(rec.symptoms),
                "drugs": len(rec.drugs),
                "tests": len(rec.tests),
            },
        }

    def tool_query_disease_knowledge(self, disease_name: str) -> dict:
        """工具：按疾病名查知识图谱（LLM 可调用版，一次查一个病）

        注意这里是「按名字查」而不是「传实体对象查」—— 因为模型只知道疾病的名字。
        KG 不可达时返回 available=False，让模型知道「没查到」而不是「没有」。
        """
        if not self.kg:
            return {
                "available": False,
                "reason": "知识图谱不可达（当前为无 KG 模式）",
                "disease": disease_name,
            }
        try:
            return {
                "available": True,
                "disease": disease_name,
                "symptoms": self.kg.get_symptoms(disease_name),
                "common_drugs": self.kg.get_common_drugs(disease_name),
                "needed_checks": self.kg.get_need_checks(disease_name),
                "complications": self.kg.get_complications(disease_name),
            }
        except Exception as e:  # 图谱异常 -> 告诉模型，让它自己决定绕过
            return {"available": False, "reason": f"图谱查询异常: {e}", "disease": disease_name}

    def tool_run_qc_rules(self, text: str) -> dict:
        """工具：运行质控规则（LLM 可调用版；内部自己抽实体，模型不用传实体）"""
        rec = self.extract_entities(text)
        findings = self.rules.run_all(text, rec, self.kg)
        return {
            "total": len(findings),
            "findings": [
                {
                    "type": f.type,
                    "severity": f.severity,
                    "description": f.description,
                    "suggestion": f.suggestion,
                    "entities": list(f.entities),
                }
                for f in findings
            ],
            "severity_summary": {
                "severe": sum(1 for f in findings if f.severity == "severe"),
                "warning": sum(1 for f in findings if f.severity == "warning"),
                "info": sum(1 for f in findings if f.severity == "info"),
            },
        }

    # ==================================================================
    # P1 新增：这三个工具才有真正的「决策空间」。
    #
    # 为什么 P0 那三个不够：extract_entities 和 run_qc_rules 是「永远要调」的，
    # query_disease_knowledge 也只有一点点选择余地。模型在这种工具集上，
    # 「规划」会退化成「按固定顺序调用」—— 形式上是 Agent，实质还是管线。
    #
    # 下面这三个都需要模型自己判断「要不要调、拿什么参数调」：
    #   · find_diseases_by_symptom —— 病历只有症状没诊断时，要不要反查可能疾病？
    #   · get_disease_detail       —— 抽到多个疾病时，先查哪个？
    #   · check_drug_indication    —— 发现用药了，要不要核对它和诊断是否相符？
    #
    # ⚠️ 前两个用的正是 src/kg/queries.py 里「早就写好、但全项目从未被调用」的
    #    get_diseases_by_symptom() 和 get_disease_info() —— 现在它们才真正被用上。
    # ==================================================================

    def tool_find_diseases_by_symptom(self, symptom: str) -> dict:
        """工具：按症状反查可能的疾病（图谱里症状→疾病的逆向查询）"""
        if not self.kg:
            return {"available": False, "reason": "知识图谱不可达（当前为无 KG 模式）",
                    "symptom": symptom}
        try:
            diseases = self.kg.get_diseases_by_symptom(symptom) or []
            result = {
                "available": True,
                "symptom": symptom,
                "possible_diseases": diseases[:30],  # 截断，避免 token 爆炸
                "count": len(diseases),
            }
            if not diseases:
                # 「0 条」有两种原因，语义完全不同：
                #   图谱根本没这个症状名（多为词典用词 ≠ 图谱用词）→ 无法反查
                #   图谱有这个症状但没关联疾病                        → 确实无关联
                # 不说清，模型会误判成「没有任何疾病有这个症状」。
                in_kg = self.kg.symptom_exists(symptom)
                result["symptom_in_kg"] = in_kg
                result["conclusion"] = (
                    f"图谱收录了症状「{symptom}」，但未关联任何疾病"
                    if in_kg else
                    f"图谱未收录症状「{symptom}」（多为实体词典用词与图谱用词不一致），"
                    "无法据此反查疾病；可换用图谱标准症状名重试，不要据此判断「该症状无对应疾病」"
                )
            return result
        except Exception as e:
            return {"available": False, "reason": f"图谱查询异常: {e}", "symptom": symptom}

    def tool_get_disease_detail(self, disease: str) -> dict:
        """工具：查询单个疾病的详情（病因 / 预防 / 治疗方式）"""
        if not self.kg:
            return {"available": False, "reason": "知识图谱不可达（当前为无 KG 模式）",
                    "disease": disease}
        try:
            info = self.kg.get_disease_info(disease)
            if not info:
                return {"available": True, "disease": disease, "found": False,
                        "note": "知识图谱中未收录该疾病"}
            # queries.py 返回的键形如 "d.name" / "d.cause"，统一剥掉前缀，方便模型读
            cleaned = {k.split(".", 1)[-1]: v for k, v in info.items()}
            cleaned["found"] = True
            return {"available": True, "disease": disease, **cleaned}
        except Exception as e:
            return {"available": False, "reason": f"图谱查询异常: {e}", "disease": disease}

    def tool_check_drug_indication(self, drug: str, disease: str) -> dict:
        """工具：核查「用药-诊断一致性」—— 这个药是不是用于治疗这个诊断？

        有独立决策价值：规则引擎只能判断「药存在、诊断存在」，
        不理解两者之间的药理学关系；模型主动调用本工具后，判断就有图谱依据了。

        ⚠️ 必须把三种情形分开说，否则会给正确处方报假警：
          · 药不在图谱里（多为通用名 vs 商品名/剂型名差异）→ unknown，不代表用药有误
          · 药在图谱里但没记录适应症                        → unknown，无法判断
          · 药在图谱里有适应症、但不含该诊断                → inconsistent，才是真值得提示的
        """
        if not self.kg:
            return {"available": False, "reason": "知识图谱不可达（当前为无 KG 模式）",
                    "drug": drug, "disease": disease}
        try:
            rows = self.kg.get_drug_diseases(drug) or []
            indicated = [r.get("disease") for r in rows if r.get("disease")]
            consistent = disease in indicated

            if consistent:
                verdict = "consistent"
                conclusion = f"图谱显示 {drug} 可用于治疗 {disease}"
            elif indicated:
                verdict = "inconsistent"
                conclusion = (
                    f"图谱显示 {drug} 的适应症不包含 {disease}"
                    f"（该药在图谱中对应 {len(indicated)} 个疾病），建议核实用药与诊断是否相符"
                )
            elif self.kg.drug_exists(drug):
                verdict = "unknown"
                conclusion = (
                    f"图谱收录了 {drug}，但未记录其适应症，无法判断与 {disease} 是否一致"
                )
            else:
                verdict = "unknown"
                conclusion = (
                    f"图谱未收录药品「{drug}」，无法判断与 {disease} 是否一致"
                    "（常见原因是通用名与图谱中的商品名/剂型名不一致，不代表用药有误，勿据此报质控问题）"
                )

            return {
                "available": True,
                "drug": drug,
                "disease": disease,
                "verdict": verdict,
                "consistent": consistent,
                "indicated_diseases": indicated[:30],
                "indicated_count": len(indicated),
                "conclusion": conclusion,
            }
        except Exception as e:
            return {"available": False, "reason": f"图谱查询异常: {e}",
                    "drug": drug, "disease": disease}