"""check_env.py —— 安装自检:通过则打印 4 行 OK"""
import os
import sys

# Windows 控制台默认 GBK,模型回复含 emoji 时 print 会抛 UnicodeEncodeError
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()  # 读取当前目录 .env

# 1. ragas 可导入(版本钉子装对了才会走到这里)
import ragas

print("OK 1/4 ragas", ragas.__version__)

# 2. 新版指标命名空间可用
from ragas.metrics.collections import Faithfulness  # noqa: F401

print("OK 2/4 ragas.metrics.collections")

# 3. langchain-openai 可用(OpenAI 兼容协议接智谱)
from langchain_openai import ChatOpenAI  # noqa: F401

print("OK 3/4 langchain_openai")

# 4. 智谱端点连通性(花 1 次极小的对话调用)
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["OPENAI_API_BASE"],
    api_key=os.environ["OPENAI_API_KEY"],
)
resp = client.chat.completions.create(
    model=os.getenv("RAGAS_JUDGE_MODEL", "glm-5.3-flash"),
    messages=[{"role": "user", "content": "回复:OK"}],
    # 思考型模型:思维链与回答共享 max_tokens,太小会被思考耗尽导致 content 为空
    # max_tokens=1024,
)
print("OK 4/4 智谱连通,回复:", resp.choices[0].message.content)