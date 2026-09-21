"""
ReAct 循环 —— 让 LLM 真正自主决定「调哪个工具、调几次、什么时候停」。

这是 P0 的核心：把原来的「固定顺序调用」换成「模型自己规划」。

四个保护（一个都不能少，否则会出问题）：
  ① MAX_STEPS 上限        —— 防死循环（模型可能反复调同一个工具，烧钱且不返回）
  ② 工具异常捕获          —— 工具失败不炸整个请求，而是把错误当作 Observation 回灌给模型
  ③ 参数校验             —— 放在 tool_registry 里，防止模型幻觉参数导致中途崩
  ④ 超步数回退 fixed 管线 —— 保证任何情况下都有结果产出（降级底线）

设计取舍（面试要点）：
  本模块是「可选增强」，不是默认路径。默认仍然是 fixed 确定性管线，原因：
    - 医疗场景要求流程可复现、可审计
    - 25 例评测集是覆盖 fixed 路径的，agent 路径非确定性、无法用精确期望值评测
  两者产出同一个 QCReport，上层 API / UI / MCP 无感。
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

from src.agent.tool_registry import ToolError, dispatch, dumps, parse_arguments, tool_names
from src.agent.tool_schema import TOOLS

# 保护①：最大步数。6 步足够「抽实体 → 查图谱 → 跑规则 → 汇总」，
# 再多说明模型在打转。这个值是可配的，不同场景可以调。
DEFAULT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "6"))

PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "prompts",
    "react_system.prompt",
)


def load_system_prompt(path: str = PROMPT_PATH) -> str:
    """读取 ReAct 系统提示词。

    Prompt 抽到 prompts/ 目录（而不是写死在 .py 里）的好处：
      改 Prompt 不用改代码、可以单独做版本管理、也方便 A/B 对比。
    """
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        # 提示词缺失不该让整个功能不可用 —— 退化成一句最简说明
        return (
            "你是病历质控助手。你可以调用工具来获取信息，"
            "先思考再决定调用哪个工具，信息足够后给出中文 Markdown 质控报告。"
        )


def run_agent(
    agent_tools,
    llm,
    text: str,
    max_steps: int = DEFAULT_MAX_STEPS,
    verbose: bool = False,
) -> dict:
    """跑一次 ReAct 循环。

    Returns:
        {
          "trace":        [ {step, thought, tool, args, observation, error}, ... ],
          "final_text":   str,     # 模型的最终回答（可能为空）
          "tool_outputs": {工具名: 最近一次结果},
          "steps":        int,     # 实际用了多少步
          "fallback":     bool,    # True = 超步数/异常，需要回退到 fixed 管线
          "stop_reason":  str,     # stop / max_steps / no_tool_calls / error
        }
    """
    if llm is None or not getattr(llm, "configured", False):
        return {
            "trace": [], "final_text": "", "tool_outputs": {},
            "steps": 0, "fallback": True, "stop_reason": "llm_unavailable",
        }

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": load_system_prompt()},
        {"role": "user", "content": text},
    ]
    trace: list[dict] = []
    tool_outputs: dict[str, Any] = {}
    final_text = ""

    for step in range(1, max_steps + 1):
        t0 = time.time()
        try:
            resp = llm.chat_with_tools(messages, tools=TOOLS, temperature=0.0, max_tokens=1200)
        except Exception as e:
            # 整个 LLM 调用失败 -> 交给上层降级（不在这里重试，重试逻辑在 llm_agent._safe_chat）
            trace.append({"step": step, "thought": "", "tool": None, "args": {},
                          "observation": None, "error": f"LLM 调用失败: {e}",
                          "elapsed_ms": int((time.time() - t0) * 1000)})
            return {"trace": trace, "final_text": "", "tool_outputs": tool_outputs,
                    "steps": step, "fallback": True, "stop_reason": "llm_error"}

        content = resp["content"]
        tool_calls = resp["tool_calls"]

        # ---- 模型不再要求调工具 => 它认为信息够了，循环结束 ----
        if not tool_calls:
            final_text = content
            trace.append({"step": step, "thought": content, "tool": None, "args": {},
                          "observation": None, "error": None,
                          "elapsed_ms": int((time.time() - t0) * 1000)})
            return {"trace": trace, "final_text": final_text, "tool_outputs": tool_outputs,
                    "steps": step, "fallback": False, "stop_reason": "stop"}

        # ---- 把助手这条（含 tool_calls）原样回填，协议要求 ----
        messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            args: dict = {}  # 先初始化：parse_arguments 可能抛错，trace 里仍要能记录
            try:
                args = parse_arguments(fn.get("arguments"))
                # 保护③：参数校验 + 执行（失败会抛 ToolError）
                result = dispatch(agent_tools, name, args)
                tool_outputs[name] = result
                payload, error = dumps(result), None
            except ToolError as e:
                # 保护②：工具失败不抛出，而是把错误当作 Observation 回灌给模型，
                # 让它自己决定「换个工具 / 换个参数 / 跳过」。这才是真正的 Agent 语义。
                payload, error = f'{{"error": "{e}"}}', str(e)

            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "content": payload,
            })
            trace.append({
                "step": step,
                "thought": content,
                "tool": name,
                "args": args,
                "observation": result if error is None else None,
                "error": error,
                "elapsed_ms": int((time.time() - t0) * 1000),
            })
            if verbose:
                print(f"  [step {step}] {name}({json.dumps(args, ensure_ascii=False)})"
                      f" -> {'ERR: ' + error if error else 'ok'}")

    # 保护④：跑到上限还没收敛 -> 标记回退，由调用方切回 fixed 管线
    return {"trace": trace, "final_text": final_text, "tool_outputs": tool_outputs,
            "steps": max_steps, "fallback": True, "stop_reason": "max_steps"}


def format_trace(trace: list[dict]) -> str:
    """把 trace 渲染成人可读的 Thought/Action/Observation 文本（面试演示用）。"""
    lines: list[str] = []
    for i, t in enumerate(trace, 1):
        head = f"[{i}] step {t['step']}  ({t['elapsed_ms']}ms)"
        lines.append(head)
        if t.get("thought"):
            thought = t["thought"].strip().replace("\n", " ")
            lines.append(f"    Thought     : {thought[:160]}")
        if t.get("tool"):
            lines.append(f"    Action      : {t['tool']}({json.dumps(t.get('args') or {}, ensure_ascii=False)})")
            if t.get("error"):
                lines.append(f"    Observation : ⚠️ {t['error']}")
            else:
                obs = json.dumps(t.get("observation"), ensure_ascii=False)
                lines.append(f"    Observation : {obs[:200]}")
        lines.append("")
    return "\n".join(lines)
