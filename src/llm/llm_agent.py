"""
LLM Agent 增强：让 LLM 扮演 Agent 的"大脑"

两个增强点：
1. extract_structured: LLM 从病历抽取结构化信息（诊断/症状/用药/检查）→ 与 NER 结果合并
2. generate_qc_report: LLM 汇总 NER + 知识图谱 + 规则发现，生成自然语言质控报告

任何一步失败：打印原因（利于排查）+ 重试 1 次；仍失败返回 None，调用方降级。
"""
import json
import sys
import time

from src.llm.client import LLMClient

EXTRACT_SYSTEM = """你是专业的医疗信息抽取引擎。从病历文本中抽取医疗实体，只输出 JSON，不要任何解释。
JSON 格式（没有的字段给空数组）：
{
  "diseases": ["诊断/疾病列表"],
  "symptoms": ["症状列表"],
  "drugs": ["用药列表"],
  "tests": ["检查项目列表"]
}"""

REPORT_SYSTEM = """你是资深病历质控专家。基于给定的病历实体、知识图谱信息和质控发现，生成一份专业的质控报告。
要求：
1. 用自然语言总结病历要点（诊断、症状、用药）
2. 对每个质控发现给出明确的问题解释和修改建议
3. 语气专业客观，疑似矛盾处用"可能""建议核实"等措辞
4. 最后给一句总体结论（合格/需修改）"""


def _safe_chat(llm: LLMClient, messages: list, **kwargs) -> str:
    """调用 LLM，失败重试 1 次，并打印原因（不吞异常到静默）"""
    for attempt in range(2):
        try:
            return llm.chat(messages, **kwargs)
        except Exception as e:
            if attempt == 1:
                print(f"[LLM] chat 调用失败(重试后): {e}", file=sys.stderr)
            else:
                print(f"[LLM] chat 调用失败，重试一次: {e}", file=sys.stderr)
                time.sleep(2)
    raise RuntimeError("LLM chat 调用失败")


def extract_structured(llm: LLMClient, text: str) -> dict | None:
    """LLM 抽取结构化病历 JSON；失败返回 None"""
    try:
        content = _safe_chat(llm, [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": text},
        ], temperature=0.0, max_tokens=600)
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end == -1:
            return None
        return json.loads(content[start:end + 1])
    except Exception as e:
        print(f"[LLM] 结构化抽取失败: {e}", file=sys.stderr)
        return None


def generate_qc_report(llm: LLMClient, text: str, entities: dict,
                       findings: list, summary: dict, kg_info: dict) -> str | None:
    """LLM 生成自然语言质控报告；失败返回 None"""
    try:
        context = json.dumps({
            "病历文本": text,
            "实体": {
                "疾病": [e.name for e in entities.diseases],
                "症状": [e.name for e in entities.symptoms],
                "药品": [e.name for e in entities.drugs],
                "检查": [e.name for e in entities.tests],
            },
            "质控发现": [
                {"级别": f.severity, "问题": f.description, "建议": f.suggestion}
                for f in findings
            ],
            "摘要": summary,
            "知识图谱参考": {k: (v if isinstance(v, list) else v) for k, v in (kg_info or {}).items()},
        }, ensure_ascii=False, indent=2)
        return _safe_chat(llm, [
            {"role": "system", "content": REPORT_SYSTEM},
            {"role": "user", "content": f"病历与质控信息如下：\n{context}"},
        ], temperature=0.3, max_tokens=1200)
    except Exception as e:
        print(f"[LLM] 质控报告生成失败: {e}", file=sys.stderr)
        return None
