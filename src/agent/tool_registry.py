"""
工具注册表 —— 把「工具名」映射到「实现」，并做参数校验。

为什么要独立一层：
  ReAct 循环只负责「编排」——它不应该知道每个工具内部怎么实现。
  新增工具时只改本文件 + tool_schema.py，循环代码一行不动。
  这是「编排」与「能力」解耦，也是 MCP 的思路：Server 提供能力，Client 负责编排。

参数校验为什么必须有：
  模型可能幻觉出不存在的参数名、漏传必填项、或把类型搞错。
  不校验就直接透传，会以莫名其妙的 TypeError 崩在中途；
  校验后能返回一个「可读的错误」给模型，让它自己修正再试。
"""
from __future__ import annotations

import json

from src.agent.tool_schema import TOOLS
from src.agent.tools import AgentTools


class ToolError(Exception):
    """工具调用失败（参数错 / 工具不存在 / 执行异常），会被回灌给模型。"""


#: 工具名 -> (AgentTools 上的方法名, 必填参数名列表)
_REGISTRY: dict[str, tuple[str, list[str]]] = {
    "extract_entities": ("tool_extract_entities", ["text"]),
    "query_disease_knowledge": ("tool_query_disease_knowledge", ["disease_name"]),
    "run_qc_rules": ("tool_run_qc_rules", ["text"]),
    # P1 新增
    "find_diseases_by_symptom": ("tool_find_diseases_by_symptom", ["symptom"]),
    "get_disease_detail": ("tool_get_disease_detail", ["disease"]),
    "check_drug_indication": ("tool_check_drug_indication", ["drug", "disease"]),
}

#: 启动时自检：schema 里的工具必须有实现，实现里的工具必须有 schema。
#  这能防止「加了 schema 忘了写实现」这种低级错误在运行时才暴露。
_SCHEMA_NAMES = {t["function"]["name"] for t in TOOLS}
if _SCHEMA_NAMES != set(_REGISTRY):
    raise RuntimeError(
        f"工具定义与实现不一致：schema={sorted(_SCHEMA_NAMES)} registry={sorted(_REGISTRY)}"
    )


def tool_names() -> list[str]:
    """当前已注册的工具名（顺序与 schema 一致，便于日志对照）。"""
    return [t["function"]["name"] for t in TOOLS]


def parse_arguments(raw: str | dict | None) -> dict:
    """把模型给的 arguments 解析成 dict。

    模型返回的是 JSON **字符串**（不是对象），且偶尔会包一层 markdown 代码块。
    这里做容错解析：解析不出来就抛 ToolError，由循环回灌给模型让它重试。
    """
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    text = str(raw).strip()
    # 容错：模型有时会输出 ```json ... ``` 包裹
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise ToolError(f"参数不是合法 JSON：{e}；收到的是：{text[:200]}") from e
    if not isinstance(parsed, dict):
        raise ToolError(f"参数必须是 JSON 对象，收到的是 {type(parsed).__name__}")
    return parsed


def validate_args(name: str, args: dict) -> None:
    """校验参数：工具是否存在、必填项是否齐全、是否传了多余参数。"""
    if name not in _REGISTRY:
        raise ToolError(f"不存在名为 {name!r} 的工具；可用工具：{tool_names()}")

    _, required = _REGISTRY[name]
    missing = [p for p in required if p not in args or args[p] in (None, "")]
    if missing:
        raise ToolError(f"工具 {name} 缺少必填参数：{missing}")

    # 多余参数：模型偶尔会自创参数名。这里只提示、不拒绝 —— 拒绝反而更容易让它卡住。
    extra = [k for k in args if k not in required]
    if extra:
        args = {k: v for k, v in args.items() if k in required}


def dispatch(agent_tools: AgentTools, name: str, args: dict) -> dict:
    """执行一个工具调用，返回 JSON 可序列化的结果。

    任何异常都被转成 ToolError —— 因为 ReAct 循环会把错误信息**当作观察结果回灌给模型**，
    让模型自己决定「换个工具」还是「跳过」。这比直接抛异常炸掉整个请求更符合 Agent 语义。
    """
    validate_args(name, args)
    method_name, required = _REGISTRY[name]
    kwargs = {k: args[k] for k in required}

    method = getattr(agent_tools, method_name, None)
    if method is None:
        raise ToolError(f"工具 {name} 的实现在 AgentTools 上不存在：{method_name}")

    try:
        result = method(**kwargs)
    except Exception as e:
        raise ToolError(f"工具 {name} 执行出错：{type(e).__name__}: {e}") from e

    if not isinstance(result, dict):
        raise ToolError(f"工具 {name} 返回值必须是 dict，实际是 {type(result).__name__}")
    return result


def dumps(result: dict) -> str:
    """把工具结果序列化成给模型看的文本（压缩 JSON，省 token）。"""
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))
