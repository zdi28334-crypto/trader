import pytest

from trader.agent.llm import Reply, ToolCall, pick_model
from trader.agent.single import ResearchAgent


class ScriptedLLM:
    """按剧本返回：先依次调三个工具，最后给出报告。并记录收到的消息。"""

    def __init__(self, script):
        self.script = list(script)
        self.seen = []

    def __call__(self, messages, tools, model):
        self.seen.append(list(messages))
        return self.script.pop(0)


def make_agent(tmp_path, price_df, llm):
    return ResearchAgent(
        chat_fn=llm,
        data_fn=lambda symbol, start, end=None: price_df,
        report_dir=tmp_path,
        verbose=False,
    )


def test_agent_runs_tools_and_writes_report(tmp_path, price_df):
    llm = ScriptedLLM(
        [
            Reply(None, [ToolCall("c1", "fetch_price_data", {"symbol": "600519", "start": "2023-01-01"})]),
            Reply(None, [ToolCall("c2", "get_indicators", {"symbol": "600519"}),
                         ToolCall("c3", "run_sma_backtest", {"symbol": "600519", "fast": 10, "slow": 30})]),
            Reply("# 600519 研究报告\n\n## 结论摘要\n测试", []),
        ]
    )
    agent = make_agent(tmp_path, price_df, llm)
    result = agent.run("研究 600519")

    assert [e["tool"] for e in result.tool_log] == ["fetch_price_data", "get_indicators", "run_sma_backtest"]
    assert result.report_path.exists()
    text = result.report_path.read_text(encoding="utf-8")
    assert "# 600519 研究报告" in text
    assert "附录：工具原始输出" in text
    assert "不构成任何投资建议" in text
    assert "![净值曲线](600519_sma10_30_" in text
    assert len(list(tmp_path.glob("*.png"))) == 1
    # 第三轮请求里应包含三条 tool 消息，且 id 对得上
    tool_msgs = [m for m in llm.seen[2] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2", "c3"]


def test_tool_error_is_returned_to_model(tmp_path, price_df):
    llm = ScriptedLLM(
        [
            Reply(None, [ToolCall("c1", "get_indicators", {"symbol": "600519"})]),
            Reply("报告", []),
        ]
    )
    agent = make_agent(tmp_path, price_df, llm)
    result = agent.run("x")
    assert "error" in result.tool_log[0]["result"]
    assert "fetch_price_data" in result.tool_log[0]["result"]["error"]


def test_unknown_tool_and_step_limit(tmp_path, price_df):
    llm = ScriptedLLM([Reply(None, [ToolCall(f"c{i}", "nope", {})]) for i in range(3)])
    agent = make_agent(tmp_path, price_df, llm)
    agent.max_steps = 3
    with pytest.raises(RuntimeError):
        agent.run("x")
    assert agent.tool_log[0]["result"] == {"error": "没有这个工具：nope"}


def test_pick_model(monkeypatch):
    for k in ("TRADER_MODEL", "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError):
        pick_model()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert pick_model().startswith("anthropic/")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    assert pick_model().startswith("deepseek/")
    monkeypatch.setenv("TRADER_MODEL", "custom/model")
    assert pick_model() == "custom/model"
