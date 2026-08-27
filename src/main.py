"""
MedGuardian 主入口
"""
import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config import (
    ENTITY_DICT_FILE, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    EXTRA_ENTITIES, LLM_ENABLED,
)
from src.ner.dict_matcher import DictMatcher
from src.ner.entity_linking import EntityLinker
from src.kg.connector import KGConnector
from src.kg.queries import KGQueries
from src.qc.rules import QCRules
from src.agent.tools import AgentTools
from src.agent.orchestrator import MedGuardianAgent
from src.llm.client import LLMClient
from src.api.routes import router, agent as api_agent

# 确保 stdout 输出 UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

app = FastAPI(
    title="MedGuardian API",
    description="智能病历结构化+质控Agent",
    version="0.1.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def init_agent():
    """初始化 Agent 组件"""
    print("=" * 50)
    print("  MedGuardian Agent 初始化中...")
    print("=" * 50)

    # 1. NER 模块
    print("\n[1/4] 加载实体词典...")
    matcher = DictMatcher(ENTITY_DICT_FILE, extra_entities=EXTRA_ENTITIES)
    linker = EntityLinker(ENTITY_DICT_FILE)

    # 2. KG 模块
    print("\n[2/4] 连接知识图谱...")
    kg = None
    connector = None
    try:
        import socket
        host = NEO4J_URI.replace("bolt://", "").split(":")[0]
        port = int(NEO4J_URI.split(":")[-1])
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((host, port))
        sock.close()
        if result == 0:
            connector = KGConnector(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
            kg = KGQueries(connector)
            count = kg.get_entity_count()
            print(f"  [OK] Neo4j 已连接")
        else:
            print(f"  [WARN] Neo4j 不可达 (端口 {port} 未开放)")
            print(f"  将使用无 KG 模式运行")
    except Exception as e:
        print(f"  [WARN] Neo4j 连接失败: {e}")
        print(f"  将使用无 KG 模式运行")

    # 3. QC 规则
    print("\n[3/4] 加载质控规则...")
    rules = QCRules()

    # 4. Agent 工具 + 调度器
    print("\n[4/4] 构建 Agent...")
    tools = AgentTools(matcher=matcher, linker=linker, kg=kg, rules=rules)
    llm = LLMClient() if LLM_ENABLED else None
    if llm:
        print(f"  [OK] LLM Agent 已启用: {llm.model} ({llm.base_url})")
    else:
        print("  [WARN] 未配置 LLM API Key → 纯规则模式（在项目根 .env 填写 LLM_API_KEY 可启用）")
    agent = MedGuardianAgent(tools, llm=llm)

    print("\n" + "=" * 50)
    print("  [OK] MedGuardian Agent 就绪！")
    print("=" * 50)
    return agent, connector if kg else None


@app.on_event("startup")
async def startup():
    global api_agent
    agent_instance, _ = init_agent()
    # 注入到路由模块
    import src.api.routes as routes
    routes.agent = agent_instance


app.include_router(router)


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)