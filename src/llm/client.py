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
