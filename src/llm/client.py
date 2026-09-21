"""
LLM API 客户端（OpenAI 兼容格式，默认 DeepSeek）

安全说明:
- API Key 从项目根 .env 读取（.env 已被 .gitignore 忽略，不会提交）
- 代码中不硬编码任何 key，日志不打印 key
"""
import os

import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_env_file():
    """把项目根 .env 加载进环境变量（简单解析，不依赖 python-dotenv）"""
    env_path = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


load_env_file()  # 模块导入时加载一次


class LLMClient:
    """OpenAI 兼容的 Chat Completions 客户端"""

    def __init__(self, api_key: str = None, base_url: str = None,
                 model: str = None, timeout: int = 120):
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL", "deepseek-chat")
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        """是否配置了有效的 key"""
        return bool(self.api_key and self.api_key.startswith("sk-"))

    def chat(self, messages: list[dict], temperature: float = 0.2,
             max_tokens: int = 1500) -> str:
        """调用 Chat Completions，返回回复文本"""
        if not self.configured:
            raise RuntimeError("LLM API Key 未配置（请在项目根 .env 中填写 LLM_API_KEY）")
        url = f"{self.base_url}/chat/completions"
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    # ==================================================================
    # 以下为 P0 新增：带工具调用的 Chat Completions。
    #
    # 与 chat() 的唯一区别：请求里多带一个 tools 参数，响应里可能不是文本
    # 而是一个「工具调用请求」。注意 chat() 保持原样不动 —— 现有两段 Prompt
    # 调用不受影响（向后兼容）。
    #
    # 关键点：模型不会真的执行任何东西，它只是「提出请求」。
    # 执行权在我们的代码里 —— 这是 Agent 的安全边界。
    # ==================================================================

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1500,
        tool_choice: str = "auto",
    ) -> dict:
        """带工具的 Chat Completions。

        Args:
            messages: OpenAI 格式的消息列表
            tools:    工具 schema 列表（见 src/agent/tool_schema.py）
            temperature: 抽取/规划类任务用 0 保稳定
            tool_choice: "auto" | "none" | "required"

        Returns:
            {
              "content": str | None,      # 模型说的文本（决定停止时才有意义）
              "tool_calls": list[dict],   # 模型要求调用的工具（原样，可直接回填进 messages）
              "finish_reason": str,       # "stop" = 它认为说完了；"tool_calls" = 它要调工具
            }
        """
        if not self.configured:
            raise RuntimeError("LLM API Key 未配置（请在项目根 .env 中填写 LLM_API_KEY）")

        url = f"{self.base_url}/chat/completions"
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        choice = resp.json()["choices"][0]
        msg = choice["message"]
        return {
            # 注意：模型要求调工具时 content 可能是 None，这里统一成 "" 方便处理
            "content": msg.get("content") or "",
            "tool_calls": msg.get("tool_calls") or [],
            "finish_reason": choice.get("finish_reason", ""),
        }
