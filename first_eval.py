"""first_eval.py —— 第一次 RAGAS 评估(新 API + 智谱裁判)"""
import asyncio
import os

from dotenv import load_dotenv

load_dotenv()  # 读当前目录 .env

from openai import AsyncOpenAI          # ragas 新版指标的底层客户端
from ragas.embeddings import OpenAIEmbeddings   # ragas 自带的现代 embedding 封装
from ragas.llms import llm_factory      # "裁判工厂":把任意 OpenAI 兼容端点包装成裁判
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

# ---------- 1. 组装裁判 ----------
client = AsyncOpenAI(
    base_url=os.environ["OPENAI_API_BASE"],   # https://open.bigmodel.cn/api/paas/v4
    api_key=os.environ["OPENAI_API_KEY"],
)
judge = llm_factory(os.getenv("RAGAS_JUDGE_MODEL", "glm-5.3-flash"), client=client)
embedder = OpenAIEmbeddings(
    model=os.getenv("RAGAS_EMBEDDING_MODEL", "embedding-3"), client=client
)

# ---------- 2. 准备一条评估数据 ----------
USER_INPUT = "H3C ER2100 enterprise router default management IP?"
RESPONSE = "The default management IP of the H3C ER2100 is 10.0.0.1."
CONTEXTS = [
    "H3C ER2100 Quick Config: the device default management IP is 192.168.1.1, "
    "default username admin, default password admin."
]

# ---------- 3. 打分 ----------
async def main() -> None:
    faithfulness = Faithfulness(llm=judge)
    r1 = await faithfulness.ascore(
        user_input=USER_INPUT, response=RESPONSE, retrieved_contexts=CONTEXTS
    )
    print("faithfulness =", r1.value)

    relevancy = AnswerRelevancy(llm=judge, embeddings=embedder)
    r2 = await relevancy.ascore(user_input=USER_INPUT, response=RESPONSE)
    print("answer_relevancy =", r2.value)

asyncio.run(main())