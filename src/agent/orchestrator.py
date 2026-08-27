"""
Agent 调度器 — ReAct 模式主流程
"""
import re
from src.agent.state import AgentState
from src.agent.tools import AgentTools
from src.qc.reporter import QCReporter
from src.models.schemas import QCReport, QCSummary, MedicalEntity
from src.llm.client import LLMClient
from src.llm.llm_agent import extract_structured, generate_qc_report


class MedGuardianAgent:
    """MedGuardian 主 Agent"""

    def __init__(self, tools: AgentTools, llm: LLMClient = None):
        self.tools = tools
        self.reporter = QCReporter()
        self.llm = llm

    def analyze(self, text: str) -> QCReport:
        """
        病历分析主流程（ReAct 循环）

        Steps:
        1. Thought: 分析文本基本信息
        2. Action: 实体识别
        3. Observation: 实体列表
        4. Thought: 需要链接标准化
        5. Action: 实体链接
        6. Observation: 标准化实体
        7. Thought: 需要 KG 验证
        8. Action: 查询知识图谱
        9. Observation: KG 结果
        10. Thought: 运行质控规则
        11. Action: 规则检查
        12. Observation: 质控发现
        13. Final: 生成报告
        """
        state = AgentState(original_text=text)

        # ---- Step 1-2: 实体识别 ----
        state.entities = self.tools.extract_entities(text)

        # ---- Step 3-4: 查询 KG ----
        if self.tools.kg:
            try:
                state.kg_results = self.tools.query_knowledge_graph(state.entities)
            except Exception:
                state.kg_results = {}  # KG 不可用时降级，不阻断主流程

        # ---- Step 5-6: 质控规则 ----
        state.findings = self.tools.run_qc_rules(text, state.entities)

        # ---- Final: 生成报告 ----
        severe = sum(1 for f in state.findings if f.severity == "severe")
        warning = sum(1 for f in state.findings if f.severity == "warning")
        info = sum(1 for f in state.findings if f.severity == "info")

        summary = QCSummary(
            total_findings=len(state.findings),
            severe_count=severe,
            warning_count=warning,
            info_count=info,
            passed=(severe == 0),
        )

        report = QCReport(
            original_text=text,
            entities=state.entities,
            findings=state.findings,
            summary=summary,
        )

        # ---- LLM Agent 增强（可选；无 key / 调用失败自动降级到纯规则模式）----
        if self.llm and self.llm.configured:
            try:
                report.llm_enabled = True
                # 1. LLM 结构化抽取，与 NER 结果合并（LLM 能抓到词典没覆盖的实体）
                llm_entities = extract_structured(self.llm, text)
                if llm_entities:
                    self._merge_llm_entities(report.entities, llm_entities)
                # 2. LLM 汇总 NER+KG+规则，生成自然语言质控报告
                report.llm_report = generate_qc_report(
                    self.llm, text, report.entities, report.findings,
                    report.summary.model_dump(), getattr(state, "kg_results", {}),
                )
            except Exception:
                report.llm_enabled = False  # 降级，不阻断主流程

        return report

    @staticmethod
    def _merge_llm_entities(record, llm_data: dict):
        """把 LLM 抽取的实体合并进结构化记录（去重）"""
        field_cat = {"diseases": "疾病", "symptoms": "症状", "drugs": "药品", "tests": "检查"}
        for field, cat in field_cat.items():
            items = getattr(record, field)
            existing = {e.name for e in items}
            for name in llm_data.get(field, []) or []:
                name = str(name).strip()
                if name and name not in existing:
                    items.append(MedicalEntity(name=name, std_name=name, category=cat))

    def analyze_markdown(self, text: str) -> str:
        """分析并返回 Markdown 格式报告"""
        report = self.analyze(text)
        return self.reporter.generate_markdown(report)