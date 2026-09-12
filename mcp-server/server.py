#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MedGuardian MCP Server — 把病历质控 Agent 暴露为 MCP 工具（stdio transport）

对应岗位 JD：基于 MCP(Model Context Protocol) 等协议开发/维护 Server，
打通电子病历、知识库等数据与工具源 —— 这里把 MedGuardian 的整体分析能力封装成一个 tool。

依赖:  pip install mcp
运行:
    python mcp-server/server.py                # stdio 模式（供 MCP 客户端连接）
    python mcp-server/server.py --selftest     # 不连客户端，直接自测一个病历，验证工具可用
客户端连接示例:
    - Claude Desktop / Cursor / 其它 MCP 客户端：添加 stdio server, command=python,
      args=[<本项目绝对路径>/mcp-server/server.py]
    - 或调试: npx @modelcontextprotocol/inspector python mcp-server/server.py

行为:
    - 无 Neo4j 时自动降级为「无 KG」模式；.env 有 LLM_API_KEY 时启用 LLM 增强，
      否则为纯规则模式 —— 任意环境都能跑出结构化质控发现。
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MedGuardian")

_agent = None


def _build_agent():
    """按 main.py init_agent 的装配逻辑构造 Agent（不引入 FastAPI/uvicorn）。
    Neo4j 不可达→降级无KG；无 LLM key→纯规则。"""
    from src.config import (
        ENTITY_DICT_FILE, EXTRA_ENTITIES, NEO4J_URI, NEO4J_USER,
        NEO4J_PASSWORD, LLM_ENABLED,
    )
    from src.ner.dict_matcher import DictMatcher
    from src.ner.entity_linking import EntityLinker
    from src.qc.rules import QCRules
    from src.agent.tools import AgentTools
    from src.agent.orchestrator import MedGuardianAgent
    from src.llm.client import LLMClient

    matcher = DictMatcher(ENTITY_DICT_FILE, extra_entities=EXTRA_ENTITIES)
    linker = EntityLinker(ENTITY_DICT_FILE)

    kg = None
    try:  # Neo4j 可达才启用 KG（import 可能打印第三方依赖告警，可忽略）
        from src.kg.connector import KGConnector
        from src.kg.queries import KGQueries
        q = KGQueries(KGConnector(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD))
        q.get_entity_count()
        kg = q
    except Exception:
        kg = None

    llm = LLMClient() if LLM_ENABLED else None
    tools = AgentTools(matcher=matcher, linker=linker, kg=kg, rules=QCRules())
    return MedGuardianAgent(tools, llm=llm)


def _get_agent():
    global _agent
    if _agent is None:
        _agent = _build_agent()
    return _agent


@mcp.tool()
def qc_analyze_emr(emr_text: str) -> str:
    """对一段中文电子病历做质控（Medical QA/质控 Agent 主能力）。

    流程：① 医疗实体识别(NER) → ② 知识图谱补全(可达时) → ③ 规则/LLM 质控 →
    ④ 生成结构化发现。检测类型：gender_mismatch(性别矛盾)、dosage_abnormal(剂量异常)、
    missing_info(信息缺失)、symptom_disease_mismatch(症状-诊断不符)。

    参数:
        emr_text: 病历原文，建议含 年龄/性别、主诉、诊断、医嘱(剂量) 等信息。

    返回: Markdown 质控报告（发现列表 + 严重度汇总 + 实体）。
    """
    return _get_agent().analyze_markdown(emr_text)


def _selftest():
    sample = ("患者，男，56岁，因下腹痛就诊。既往：2型糖尿病5年，口服二甲双胍500mg每日三次。"
              "影像学：子宫肌瘤，大小约5cm。医嘱：己烯雌酚 2mg 每日一次。空腹血糖 9.8mmol/L。")
    print(_get_agent().analyze_markdown(sample))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="直接自测一个病历（不启动 MCP）")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        mcp.run()
