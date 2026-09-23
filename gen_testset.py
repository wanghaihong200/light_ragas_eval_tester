"""gen_testset.py —— 从纯文本切片自动生成评估集(实测:ragas 0.4.3 + 智谱)"""
import os

from dotenv import load_dotenv

load_dotenv()

from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.testset import TestsetGenerator

client = AsyncOpenAI(
    base_url=os.environ["OPENAI_API_BASE"], api_key=os.environ["OPENAI_API_KEY"]
)

# ⚠️ 注意:这里必须用新组件 llm_factory / OpenAIEmbeddings。
# 实测用旧封装 LangchainLLMWrapper 会报
# AttributeError: 'LangchainLLMWrapper' object has no attribute 'agenerate_prompt'
generator = TestsetGenerator(
    llm=llm_factory(os.getenv("RAGAS_JUDGE_MODEL", "glm-5.3-flash"), client=client),
    embedding_model=OpenAIEmbeddings(
        model=os.getenv("RAGAS_EMBEDDING_MODEL", "embedding-3"), client=client
    ),
)

# 直接喂文本切片(等价于你知识库切分后的 chunk;真实项目里从 Milvus/文档导出)
chunks = [
    "H3C ER2100 企业级路由器默认管理地址为 192.168.1.1,默认用户名和密码均为 admin,"
    "首次登录后必须修改默认口令。",
    "ER2100 支持 WAN 口聚合,配置路径为 网络 > 接口配置 > WAN 聚合,"
    "聚合后总带宽为各物理口之和,需重启设备生效。",
]

testset = generator.generate_with_chunks(chunks, testset_size=2)
df = testset.to_pandas()
df.to_csv("auto_testset.csv", index=False, encoding="utf-8")
print(df[["user_input", "reference"]].to_string())