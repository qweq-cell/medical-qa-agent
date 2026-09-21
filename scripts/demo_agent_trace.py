#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P0 演示：跑一次 ReAct 循环，打印完整的 Thought / Action / Observation。

这是 P0 的验收脚本 —— 能在这里看到「模型自己决定调用顺序」，就说明改造成功。

用法:
    python scripts/demo_agent_trace.py                      # 用内置样本病历
    python scripts/demo_agent_trace.py --text "患者，男，..."   # 用自己的病历
    python scripts/demo_agent_trace.py --no-llm             # 只验证工具链路（不调 LLM）
    python scripts/demo_agent_trace.py --json               # 输出 JSON（便于对比多次运行）

前置:
    .env 里配好 LLM_API_KEY（DeepSeek）。没配也能跑，但只会走「工具链路自检」。
"""
import argparse
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# Windows 控制台默认 GBK，报告里有 emoji 会崩 —— 统一成 UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

SAMPLE_EMR = (
    "患者，男，56岁，因下腹痛3月就诊。既往：2型糖尿病5年，口服二甲双胍500mg每日三次。"
    "影像学：子宫肌瘤，大小约5cm。医嘱：己烯雌酚 2mg 每日一次。空腹血糖 9.8mmol/L。"
)


def build_agent_tools():
    """按 main.py 的装配逻辑构造 AgentTools（不引入 FastAPI）。"""
    from src.config import (
        ENTITY_DICT_FILE, EXTRA_ENTITIES, NEO4J_URI, NEO4J_USER,
        NEO4J_PASSWORD, LLM_ENABLED,
    )
    from src.ner.dict_matcher import DictMatcher
    from src.ner.entity_linking import EntityLinker
    from src.qc.rules import QCRules
    from src.agent.tools import AgentTools

    matcher = DictMatcher(ENTITY_DICT_FILE, extra_entities=EXTRA_ENTITIES)
    linker = EntityLinker(ENTITY_DICT_FILE)

    kg = None
    try:  # Neo4j 可达才启用 KG；不可达自动降级为无图谱模式
        from src.kg.connector import KGConnector
        from src.kg.queries import KGQueries
        q = KGQueries(KGConnector(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD))
        q.get_entity_count()
        kg = q
        print("[KG] Neo4j 已连接")
    except Exception as e:
        print(f"[KG] Neo4j 不可达，降级为无图谱模式（{type(e).__name__}）")

    from src.llm.client import LLMClient
    llm = LLMClient() if LLM_ENABLED else None
    print(f"[LLM] {'已启用: ' + llm.model if llm else '未配置 LLM_API_KEY（仅验证工具链路）'}")

    return AgentTools(matcher=matcher, linker=linker, kg=kg, rules=QCRules()), llm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=SAMPLE_EMR, help="病历文本")
    ap.add_argument("--max-steps", type=int, default=6, help="最大步数（防死循环）")
    ap.add_argument("--no-llm", action="store_true", help="只验证工具链路，不调 LLM")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()

    print("=" * 72)
    print("  MedGuardian · ReAct Agent Trace")
    print("=" * 72)
    print(f"\n病历：{args.text[:60]}...\n")

    agent_tools, llm = build_agent_tools()

    from src.agent.react import format_trace, run_agent
    from src.agent.tool_registry import dispatch, tool_names

    if args.no_llm:
        print("\n--- 工具链路自检（不调 LLM）---")
        for name, call_args in [
            ("extract_entities", {"text": args.text}),
            ("run_qc_rules", {"text": args.text}),
        ]:
            r = dispatch(agent_tools, name, call_args)
            print(f"  {name}: {json.dumps(r, ensure_ascii=False)[:220]}")
        print(f"\n可用工具：{tool_names()}")
        return

    print(f"\n可用工具：{tool_names()}\n")
    print("--- ReAct 循环 ---")
    result = run_agent(agent_tools, llm, args.text,
                       max_steps=args.max_steps, verbose=True)

    print("\n" + "=" * 72)
    print(f"  步数: {result['steps']} | 停止原因: {result['stop_reason']} | 回退: {result['fallback']}")
    print("=" * 72)

    print("\n--- 完整 Trace ---")
    print(format_trace(result["trace"]))

    if result["final_text"]:
        print("--- 模型最终回答 ---")
        print(result["final_text"])

    # 工具调用顺序 —— 这就是「模型自主规划」的证据
    called = [t["tool"] for t in result["trace"] if t.get("tool")]
    print("\n--- 工具调用顺序（模型自己决定的）---")
    print("  " + " → ".join(called) if called else "  (没有调用任何工具)")

    if args.json:
        print("\n--- JSON ---")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
