"""
Agent 工具的工具定义（Tool Schema）—— 给 LLM 看的「说明书」。

设计原则（面试要点）：
1. **参数只用字符串**。LLM 能产生的是文本，不能产生 Python 对象，
   所以每个工具都必须能用「字符串进、结构化出」的方式调用。
   这也是 MCP / Function Calling 工具的通用形态：无状态、参数简单、可序列化。
2. **description 里必须写清「什么时候该调用」**，而不是只写「这个工具干什么」。
   模型是靠 description 决定调用时机的 —— 写不好，它就不会在对的时候调。
3. **strict 模式**（可选）：DeepSeek 的 beta 接口支持 `strict: true`，
   要求 additionalProperties=false 且所有属性都列进 required，
   模型会保证输出符合 schema，能省掉大量参数校验代码。

本文件共 6 个工具：P0 三个（实体识别 / 图谱查询 / 规则校验）
+ P1 三个（按症状反查疾病 / 疾病详情 / 用药-诊断一致性）。
"""
from __future__ import annotations

#: 工具名 -> 中文别名（仅用于日志/展示，方便排查）
TOOL_LABELS = {
    "extract_entities": "实体识别",
    "query_disease_knowledge": "图谱查询",
    "run_qc_rules": "规则校验",
    # P1 新增
    "find_diseases_by_symptom": "症状反查疾病",
    "get_disease_detail": "疾病详情",
    "check_drug_indication": "用药-诊断一致性",
}


TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "extract_entities",
            "description": (
                "从中文病历文本中提取医疗实体（疾病 / 症状 / 药品 / 检查 / 科室），"
                "并做标准化（例如把「心梗」归一到「急性心肌梗死」）。"
                "这是质控流程的第一步：任何分析都应先调用它，否则无法知道病历里有什么。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "病历原文（完整传入，不要截断）",
                    }
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_disease_knowledge",
            "description": (
                "查询某个疾病的医学知识图谱信息，返回：典型症状、常用药、需要做的检查、并发症。"
                "当你需要判断「病历描述与该诊断是否相符」时调用。"
                "病历里有多个疾病时，优先查与主诉最相关的那个，不必每个都查（注意 token 成本）。"
                "注意：如果病历里只有症状、没有明确诊断，不要调用本工具（你没有疾病名可查）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "disease_name": {
                        "type": "string",
                        "description": "标准化后的疾病名称，如「2型糖尿病」「高血压」",
                    }
                },
                "required": ["disease_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_qc_rules",
            "description": (
                "运行确定性质控规则引擎，检测：性别矛盾（男性+妇科诊断等）、剂量异常（>1000mg）、"
                "信息缺失（缺年龄/诊断/剂量）、症状-诊断匹配。"
                "返回结构化发现，每条含类型、严重程度（severe/warning/info）、描述、建议。"
                "结束分析前应当调用一次，以获得确定性、可审计的质控发现。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "病历原文（与 extract_entities 传同一份）",
                    }
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_diseases_by_symptom",
            "description": (
                "根据症状反查可能对应的疾病（知识图谱的逆向查询）。"
                "当病历中出现了症状、但没有明确诊断，或需要做鉴别诊断时调用。"
                "注意：如果病历里已经有清晰诊断，一般不需要调它。"
                "本工具会自动做名称回退匹配（实体词典用词与图谱用词常不一致）："
                "match_type=exact 精确命中，fuzzy 表示已按最接近的标准症状"
                "（见 matched_symptom）反查成功，none 才表示确实没找到。"
                "只有 match_type=none 时才可以说「图谱中没有该症状」，"
                "**不要**据此认为「该症状没有对应疾病」。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symptom": {
                        "type": "string",
                        "description": "标准化后的症状名，如「口渴多饮」「头痛」",
                    }
                },
                "required": ["symptom"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_disease_detail",
            "description": (
                "查询某个疾病的详细信息：病因(cause)、预防(prevent)、治疗方式(cure_way)。"
                "当需要判断「病历描述与该疾病是否相符」时调用。"
                "疾病较多时应优先查与主诉最相关的那个，不必每个都查。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "disease": {
                        "type": "string",
                        "description": "标准化后的疾病名，如「2型糖尿病」",
                    }
                },
                "required": ["disease"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_drug_indication",
            "description": (
                "核查「用药-诊断一致性」：这个药在知识图谱里是不是用于治疗这个诊断？"
                "当病历中同时出现药品和诊断时调用，用于发现「诊断与用药不符」的质控问题。"
                "本工具会自动做名称回退匹配（图谱多为商品名/剂型名，词典给的是通用名）："
                "match_type=exact 精确命中，fuzzy 表示已按 matched_drug 匹配成功。"
                "返回 verdict（consistent / inconsistent / unknown）、consistent 布尔值、"
                "以及该药在图谱中对应的适应症列表。"
                "重要：verdict=unknown 只代表「图谱未收录该药或未记录适应症」，"
                "属于「无法判断」，**绝不能**据此判定用药有误或生成质控问题。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "drug": {"type": "string", "description": "标准化后的药品名，如「阿司匹林」"},
                    "disease": {"type": "string", "description": "标准化后的疾病名，如「胃溃疡」"},
                },
                "required": ["drug", "disease"],
            },
        },
    },
]


def tool_names() -> list[str]:
    """返回全部工具名（用于日志与校验）。"""
    return [t["function"]["name"] for t in TOOLS]
