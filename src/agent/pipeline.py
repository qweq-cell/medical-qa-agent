"""
统一分析入口 —— 按 PIPELINE_MODE 在两条管线之间切换（P2）。

两条管线的取舍（面试要点）：
  fixed  MedGuardianAgent 固定顺序：抽实体 → 查图谱 → 跑规则 → 汇总
         · 流程可复现、可审计；25 例评测集用精确期望值覆盖它
         · 医疗质控的默认路径
  agent  ReAct 循环，模型自己决定「调哪个工具、调几次、什么时候停」
         · 只有工具多才有价值：6 个工具里有 4 个需要判断要不要调
         · 非确定性，无法用精确期望值评测，所以是可选增强而非默认

两者都产出同一个 QCReport，上层 API / UI / MCP 完全无感。

为什么 agent 模式仍然先跑一遍确定性内核：
  医疗质控的 findings 必须可审计 —— 模型写的自然语言报告负责「解释」，
  规则引擎产出的 findings 才是「判定依据」。而且模型超步数/报错要能立刻回退，
  内核先算好，回退就没有额外代价。
"""
from __future__ import annotations

from src.agent.orchestrator import MedGuardianAgent
from src.agent.react import run_agent
from src.agent.tools import AgentTools
from src.config import PIPELINE_MODE
from src.llm.client import LLMClient
from src.models.schemas import QCReport

VALID_MODES = ("fixed", "agent")


def normalize_mode(mode: str | None = None) -> str:
    """把任意输入规整成合法模式；非法值退回 fixed（不因拼错就让服务挂掉）。"""
    value = (mode or PIPELINE_MODE or "fixed").strip().lower()
    return value if value in VALID_MODES else "fixed"


class MedicalPipeline:
    """统一入口：`analyze(text) -> QCReport`。"""

    def __init__(self, tools: AgentTools, llm: LLMClient = None, mode: str | None = None):
        self.tools = tools
        self.llm = llm
        self.mode = normalize_mode(mode)
        # 带 LLM 的 fixed 管线（LLM 增强：结构化抽取 + 自然语言报告）
        self._fixed_llm = MedGuardianAgent(tools, llm=llm)
        # 纯确定性内核（不调 LLM）—— agent 模式的 findings 由它产出
        self._core = MedGuardianAgent(tools, llm=None)

    # ---- 对外接口（与 MedGuardianAgent 一致，便于替换）----

    def analyze(self, text: str) -> QCReport:
        if self.mode == "agent":
            return self._analyze_agent(text)
        report = self._fixed_llm.analyze(text)
        report.pipeline_mode = "fixed"
        return report

    def analyze_markdown(self, text: str) -> str:
        report = self.analyze(text)
        return self._fixed_llm.reporter.generate_markdown(report)

    # ---- agent 管线 ----

    def _analyze_agent(self, text: str) -> QCReport:
        """ReAct 自主规划；失败 / 超步数 / 无 LLM 时自动回退 fixed。"""
        result = run_agent(self.tools, self.llm, text)

        if result.get("fallback") or not result.get("final_text"):
            # 保护④落地：回退到 fixed（此时仍可用 LLM 增强自然语言报告），
            # 并在报告里标明这次是回退来的，便于排查「为什么这次没有 trace」。
            report = self._fixed_llm.analyze(text)
            report.pipeline_mode = "agent_fallback"
            return report

        report = self._core.analyze(text)
        report.pipeline_mode = "agent"
        report.llm_enabled = True
        report.llm_report = result["final_text"]
        return report
