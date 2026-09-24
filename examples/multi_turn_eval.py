"""multi_turn_eval.py —— 多轮对话评测示例(ragas 0.4.3 新版 collections API)

多轮指标不经过旧版 MultiTurnSample 载体,直接接收消息列表
(HumanMessage / AIMessage / ToolMessage):

    TopicAdherence      对话是否围绕既定主题(LLM 裁判,mode=precision/recall/f1)
    AgentGoalAccuracy   是否达成用户目标,对照理想答案(LLM 裁判)
    ToolCallF1          工具调用名/参数与期望的 P/R/F1(免 LLM,确定性计算)

用法:
    python examples/multi_turn_eval.py
"""
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")  # .env 在仓库根目录

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from ragas.metrics.collections import AgentGoalAccuracy, ToolCallF1, TopicAdherence

# ---------- 模拟一段带工具调用的多轮对话 ----------
# 轮次 1:用户问管理地址 → Agent 调知识库检索工具 → 基于工具结果回答
# 轮次 2:追问默认密码 → Agent 依据已检索内容直接回答
CONVERSATION = [
    HumanMessage(content="H3C ER2100 路由器的默认管理地址是什么?"),
    AIMessage(
        content="",
        tool_calls=[ToolCall(name="query_kb", args={"query": "ER2100 默认管理地址"})],
    ),
    ToolMessage(
        content="H3C ER2100 企业级路由器默认管理地址为 192.168.1.1,"
                "默认用户名和密码均为 admin,首次登录后必须修改默认口令。"
    ),
    AIMessage(content="ER2100 的默认管理地址是 192.168.1.1。"),
    HumanMessage(content="那默认密码是什么?需要改吗?"),
    AIMessage(content="默认密码是 admin,与用户名相同;"
                      "出于安全考虑,首次登录后必须修改默认口令。"),
]

REFERENCE = (
    "ER2100 默认管理地址为 192.168.1.1,默认用户名和密码均为 admin,"
    "首次登录后必须修改默认口令。"
)
REFERENCE_TOPICS = ["ER2100 路由器配置"]
REFERENCE_TOOL_CALLS = [ToolCall(name="query_kb", args={"query": "ER2100 默认管理地址"})]


async def main() -> None:
    client = AsyncOpenAI(
        base_url=os.environ["OPENAI_API_BASE"], api_key=os.environ["OPENAI_API_KEY"],
    )
    judge = llm_factory(os.getenv("RAGAS_JUDGE_MODEL", "glm-5.3-flash"), client=client)

    topic = await TopicAdherence(llm=judge, mode="f1").ascore(
        user_input=CONVERSATION, reference_topics=REFERENCE_TOPICS,
    )
    goal = await AgentGoalAccuracy(llm=judge).ascore(
        user_input=CONVERSATION, reference=REFERENCE,
    )
    # ToolCallF1 是确定性指标,不需要裁判 LLM
    tool = await ToolCallF1().ascore(
        user_input=CONVERSATION, reference_tool_calls=REFERENCE_TOOL_CALLS,
    )

    print("\n===== 多轮对话评测结果 =====")
    print(f"topic_adherence   (主题贴合, f1)  = {topic.value}")
    print(f"agent_goal_accuracy(目标达成)    = {goal.value}")
    print(f"tool_call_f1      (工具调用 F1)  = {tool.value}")

if __name__ == "__main__":
    asyncio.run(main())
