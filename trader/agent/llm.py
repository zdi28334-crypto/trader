"""大模型调用：用 LiteLLM 统一封装，DeepSeek 和 Claude 只差一个模型名。

配置（写在项目根目录的 .env 文件里）：
    DEEPSEEK_API_KEY=sk-...          # 用 DeepSeek 时填
    ANTHROPIC_API_KEY=sk-ant-...     # 用 Claude 时填
    TRADER_MODEL=deepseek/deepseek-chat   # 可选，手动指定模型
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

DEFAULT_DEEPSEEK = "deepseek/deepseek-chat"
DEFAULT_CLAUDE = "anthropic/claude-sonnet-5-5"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Reply:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


def pick_model() -> str:
    """优先用 TRADER_MODEL；否则看配了哪个 key。"""
    if m := os.environ.get("TRADER_MODEL"):
        return m
    if os.environ.get("DEEPSEEK_API_KEY"):
        return DEFAULT_DEEPSEEK
    if os.environ.get("ANTHROPIC_API_KEY"):
        return DEFAULT_CLAUDE
    raise RuntimeError(
        "没有找到大模型的 API key。请在项目根目录建 .env 文件，"
        "写入 DEEPSEEK_API_KEY=... 或 ANTHROPIC_API_KEY=...（参考 .env.example）"
    )


def chat(messages: list[dict], tools: list[dict], model: str) -> Reply:
    """调一次大模型，返回文字回答或工具调用请求。"""
    import litellm

    resp = litellm.completion(model=model, messages=messages, tools=tools, temperature=0.2)
    msg = resp.choices[0].message
    calls = []
    for tc in msg.tool_calls or []:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
    return Reply(content=msg.content, tool_calls=calls)
