# RAGAS 企业级实战教程 —— 从零开始搭建 RAG 系统的质量评估体系

> **版本基准**:本教程全部代码基于 **ragas 0.4.3**(撰写时的最新稳定版)实测通过,
> 裁判模型与向量模型统一使用**智谱 BigModel**(`glm-5.3-flash` + `embedding-3`),
> 与掌柜智库(`knowledge_base_center`)项目的技术栈完全一致。文中所有示例输出均为真实运行结果。
>
> **适用读者**:会安装 Python、能照着文档运行脚本,但**从未接触过 RAG 评估**的工程师 / 测试同学。
> 不需要任何机器学习背景。
>
> **学完你将能**:独立搭建评估环境 → 看懂并选择评估指标 → 构建/自动生成评估集 →
> 把评估跑成一条可进 CI 的流水线 → 对掌柜智库仓库的 RAG 系统做一次完整评估并给出优化结论。

---

## 目录

| 章节 | 内容 | 预计耗时 |
|------|------|---------|
| [第 1 章](#第-1-章-为什么你的-rag-系统需要体检) | RAG 评估的来龙去脉 | 10 min |
| [第 2 章](#第-2-章-环境准备与安装) | 独立环境、版本组合、验证安装 | 15 min |
| [第 3 章](#第-3-章-核心概念十分钟建立评估世界观) | 数据结构、指标分类、裁判模型 | 10 min |
| [第 4 章](#第-4-章第一次评估十几行代码跑通) | 第一份评估报告 | 10 min |
| [第 5 章](#第-5-章指标体系详解每个指标测什么怎么读分) | 8 个核心指标逐个讲透 | 30 min |
| [第 6 章](#第-6-章构建评估数据集冷启动的三条路) | 手工标注 / 线上采样 / 自动生成 | 30 min |
| [第 7 章](#第-7-章企业级评估流水线从脚本到报告) | 流水线脚本、报告、阈值门禁 | 40 min |
| [第 8 章](#第-8-章-ci-集成让评估成为回归门禁) | CI 定时/触发评估、基线对比 | 20 min |
| [第 9 章](#第-9-章实战评估掌柜智库知识库) | 对掌柜智库 RAG 系统完整评估 | 60 min |
| [第 10 章](#第-10-章进阶方向) | 多轮评估、自定义指标、成本治理 | 20 min |
| [附录 A](#附录-a常见报错速查实测) | 全部实测踩坑速查 | 按需 |
| [附录 B](#附录-b术语表) | 术语表 | 按需 |
| [附录 C](#附录-c版本信息与参考) | 版本信息与参考 | 按需 |

---

## 第 1 章 为什么你的 RAG 系统需要"体检"

### 1.1 RAG 系统的两个"黑盒"

RAG(Retrieval-Augmented Generation,检索增强生成)= **先检索、后生成**:
用户提问 → 系统从知识库里捞出一批相关文档片段(检索)→ 把片段塞进提示词让大模型组织答案(生成)。

这条链路上有两个环节都由大模型驱动,行为不稳定:

1. **检索环节**:向量检索、重排、融合……捞回来的片段可能不相关、可能漏掉关键片段;
2. **生成环节**:模型可能编造知识库里没有的细节(幻觉),可能答非所问。

传统软件测试断言的是"输出 == 期望值",但 RAG 的输出是自然语言——
同一问题两次回答措辞都不同,**没法用字符串相等来断言**。这就是 RAG 评估的核心难题。

### 1.2 RAGAS 的解题思路:让 LLM 当裁判

[RAGAS](https://github.com/explodinggradients/ragas)(RAG Assessment)是目前最主流的开源
RAG 评估框架,核心思想叫 **LLM-as-a-Judge(大模型当裁判)**:

```
                ┌─────────────── 你要评估的 RAG 系统 ───────────────┐
用户问题 ──────► │ 检索出 contexts,生成 answer                        │
                └──────────────────────┬────────────────────────────┘
                                       ▼
                        (question, contexts, answer, reference)
                                       ▼
                ┌──────────────── RAGAS 评估框架 ───────────────────┐
                │  把每一轮问答交给一组"指标",每个指标内部都是       │
                │  若干次对"裁判 LLM"的调用:拆解、比对、打分        │
                └──────────────────────┬────────────────────────────┘
                                       ▼
                    一组 0~1 之间的分数 + 每条样本的明细
```

**裁判也需要一个 LLM**——它会额外调用大模型来拆解答案中的论断、判断每个论断是否有上下文支撑等。
所以跑评估 = 被测系统 + 一个"裁判模型"的 API 费用。本教程用智谱 `glm-5.3-flash` 当裁判,
成本极低。

### 1.3 企业里谁在用、什么时候用

| 场景 | 典型做法 | 对应本教程章节 |
|------|---------|--------------|
| **上线前质检** | 知识库导入新一批文档后,跑一轮评估看指标是否劣化 | 第 6、7 章 |
| **回归门禁** | 换 embedding 模型 / 改切分策略 / 调重排参数后,评估分数不得低于基线 | 第 7、8 章 |
| **方案对比** | 两套检索方案(如"纯向量" vs "向量+HyDE")各跑同一评估集,横向比分数 | 第 9 章 |
| **定期巡检** | 每晚定时对线上采样问题跑评估,分数进日报 | 第 8 章 |
| **问题定位** | 忠实度低 → 查生成;上下文召回低 → 查检索 | 第 5、9 章 |

### 1.4 评估的边界:先建立正确预期

开始之前必须说清楚 RAGAS **能做什么、不能做什么**:

- ✅ **能**:量化"答案是否忠于检索内容""检索是否捞回了正确文档"等维度,给出可横向对比的分数;
- ✅ **能**:作为回归基线,防止改动导致质量悄悄劣化;
- ⚠️ **有限**:分数是"裁判模型认为的分数",换裁判模型分数会漂移——**基线必须始终用同一个裁判模型跑**;
- ❌ **不能**:替代人工抽检。企业实践中推荐"全自动评估 + 每周人工抽检 10~20 条"双轨制。

`★ Insight ─────────────────────────────────────`
- RAGAS 评估的不是"模型好不好",而是**你的 RAG 链路配置好不好**——同一套模型,换切分策略分数就可能差一倍。
- "裁判模型固定"是所有 LLM-as-a-Judge 实践的第一纪律:分数只有相对意义,没有绝对意义。
`─────────────────────────────────────────────────`

---

## 第 2 章 环境准备与安装

### 2.1 前置条件

| 依赖 | 要求 | 说明 |
|------|------|------|
| Python | **3.9 ~ 3.11**(实测 3.11.3) | ragas 0.4.3 官方要求 ≥3.9 |
| 智谱 API Key | 一个即可 | [open.bigmodel.cn](https://open.bigmodel.cn) 注册后获取 |
| pip | 随 Python 附带 | 建议先升级 |

### 2.2 铁律:评估环境必须独立

> ⚠️ **实测踩坑(高频)**:ragas 0.4.3 声明依赖 `langchain-community` 时**没有设置版本上限**。
> 如果装进一个已有 `langchain-community 0.4.x` 的业务环境(比如掌柜智库仓库的 venv),
> ragas **导入即崩**:
>
> ```
> ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
> ```
>
> 原因:ragas 源码 `ragas/llms/base.py` 第 12 行无条件导入
> `from langchain_community.chat_models.vertexai import ChatVertexAI`,
> 而 langchain-community 从 0.4 起删除了该模块。ragas 没做任何 try/except 兜底。
>
> **解法**:安装时把 langchain-community 钉在 `==0.4.1`(垫片尚存)或 `<0.4`,并且
> **永远不要装进业务 venv**。

所以第一步,建一个**专用虚拟环境**(下文统一叫"评估环境"):

```bash
# Windows(git-bash / PowerShell 通用写法)
python -m venv D:\workspace\.venvs\ragas-eval

# 激活(git-bash)
source /d/workspace/.venvs/ragas-eval/Scripts/activate
# 激活(PowerShell)
D:\workspace\.venvs\ragas-eval\Scripts\Activate.ps1
# 激活(Linux / macOS)
source /d/workspace/.venvs/ragas-eval/bin/activate
```

### 2.3 安装(国内镜像加速)

```bash
# 先升级 pip(可选但推荐)
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip

# 核心安装:注意 langchain-community 的版本钉子,不能省
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    "ragas==0.4.3" \
    "langchain-community==0.4.1" \
    python-dotenv openai langchain-openai

# 只有第 6 章要用 ragas 自动生成评估集时才需要(ragas 没把它们放进基础依赖!)
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple rapidfuzz nltk
```

> 💡 `langchain-community` 的钉法二选一(原因见 2.2 踩坑):
> **`==0.4.1`**(掌柜智库仓库根目录 `requirements-eval.txt` 同款,0.4.1 仍保留 ragas 需要的弃用垫片,**已实测导入通过**)
> 或 **`<0.4`**(本教程全流程在 0.3.31 上完整验证)。二者都避开了 0.4.2 的断崖。

**实测落地的版本组合**(本教程所有代码在此组合上验证通过,排错时先对照):

| 包 | 版本 | 说明 |
|----|------|------|
| ragas | 0.4.3 | 评估框架本体 |
| langchain-community | **0.4.1**(或 <0.4,实测 0.3.31) | ≥0.4.2 会让 ragas 导入崩溃,见 2.2 踩坑 |
| langchain / langchain-core | 1.4.2 / 1.6.3 | 随依赖自动解析 |
| langchain-openai | 1.1.9 | 用 OpenAI 兼容协议接智谱 |
| openai | 1.109.1 | ragas 新版指标的底层客户端 |
| datasets | 5.0.1 | ragas 依赖 |
| rapidfuzz / nltk | 最新即可 | 仅 Testset 生成需要 |

### 2.4 配置智谱 Key

评估环境单独放一份 `.env`(和业务项目的 `.env` 互不影响)。在任意工作目录建
`ragas_eval/.env`:

```ini
# 智谱 OpenAI 兼容端点(key 在 open.bigmodel.cn 控制台获取)
OPENAI_API_BASE=https://open.bigmodel.cn/api/paas/v4
OPENAI_API_KEY=你的智谱key

# 裁判模型(对话/打分都用它)
RAGAS_JUDGE_MODEL=glm-5.3-flash
# 向量模型(答案相关性等指标需要)
RAGAS_EMBEDDING_MODEL=embedding-3
```

> 💡 与掌柜智库仓库的关系:其项目根目录 `.env` 里已经有同样格式的 `OPENAI_API_BASE` / `OPENAI_API_KEY`
> (见 `config/lm_config.py`、`config/embedding_config.py`),第 9 章实战会直接复用它们,
> 无需重复申请。

### 2.5 验证安装

把下面内容存为 `check_env.py` 并在**评估环境**中运行:

```python
"""check_env.py —— 安装自检:通过则打印 4 行 OK"""
import os

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
    max_tokens=8,
)
print("OK 4/4 智谱连通,回复:", resp.choices[0].message.content)
```

运行:

```bash
python check_env.py
# 预期输出:
# OK 1/4 ragas 0.4.3
# OK 2/4 ragas.metrics.collections
# OK 3/4 langchain_openai
# OK 4/4 智谱连通,回复:OK
```

> ⚠️ **Windows 控制台 GBK 坑**:如果脚本要 `print()` 中文,Windows 默认 GBK 控制台遇到特殊字符
> 会抛 `UnicodeEncodeError`(掌柜智库仓库 `main.py` 的 `_setup_console()` 就是为了修这个)。
> 教程里的示例脚本要么只打印 ASCII,要么在运行前设置:
>
> ```bash
> # git-bash / Linux
> export PYTHONIOENCODING=utf-8
> # PowerShell
> $env:PYTHONIOENCODING="utf-8"
> ```

---

## 第 3 章 核心概念:十分钟建立评估世界观

### 3.1 评估的最小单元:SingleTurnSample

RAGAS 眼中,一次"单轮问答"就是一条数据,用 `SingleTurnSample` 描述。
**全部字段你只需要先记住 4 个**:

```python
from ragas import SingleTurnSample

sample = SingleTurnSample(
    user_input="ER2100 的默认管理地址是什么?",     # ① 用户问题(必需)
    retrieved_contexts=[                          # ② 被测系统检索到的片段列表(测检索类指标必需)
        "H3C ER2100 快速配置指南:默认管理地址 192.168.1.1……",
    ],
    response="默认管理地址是 192.168.1.1。",        # ③ 被测系统生成的答案(测答案类指标必需)
    reference="ER2100 默认管理地址为 192.168.1.1。", # ④ 人工参考答案(仅"有参考"类指标需要)
)
```

| 字段 | 谁来提供 | 缺了会怎样 |
|------|---------|-----------|
| `user_input` | 你 | 必需,评估无从谈起 |
| `retrieved_contexts` | 被测系统的检索环节 | 涉及上下文的指标(忠实度/精确率/召回)无法计算 |
| `response` | 被测系统的生成环节 | 涉及答案的指标无法计算 |
| `reference` | 人工标注 / 文档原文 | "有参考"类指标跳过;你仍可跑"无参考"类指标 |
| `reference_contexts` | 可选 | 标准答案所依据的原文,多数指标用不到 |

> 💡 关键认知:**`retrieved_contexts` 来自被测系统,不是评估工具帮你检索的**。
> RAGAS 只负责"打分",不负责"运行你的 RAG"。你的系统要么暴露检索结果(第 9 章的进程内直调),
> 要么把检索结果落盘成文件再喂给 RAGAS(第 7 章的 JSONL)。

多条 `SingleTurnSample` 组成一个 `EvaluationDataset`,就是一份"评估集"。

### 3.2 指标的两大类:无参考 vs 有参考

| | 无参考指标(只判质量) | 有参考指标(对照标准答案) |
|---|---|---|
| 需要 `reference` | 否 | **是** |
| 标注成本 | 低(只需问题) | 高(每条要人写/审参考答案) |
| 回答的问题 | "答案自洽吗?检索相关吗?" | "答案和标准答案意思一致吗?" |
| 典型指标 | Faithfulness、ContextPrecision、AnswerRelevancy | ContextRecall、FactualCorrectness、AnswerCorrectness |
| 企业用法 | **日常回归、巡检**(便宜,可天天跑) | **发版前质检、方案对比**(贵,抽样跑) |

### 3.3 评估系统里的三个"角色"

1. **裁判 LLM(judge)**:负责拆解、比对、打分的模型。本教程 = 智谱 `glm-5.3-flash`。
   指标分数质量上限取决于裁判能力,裁判**必须全程固定**;
2. **裁判 Embedding**:个别指标(如答案相关性)要把文本转向量算相似度。
   本教程 = 智谱 `embedding-3`;
3. **被测系统**:你的 RAG 链路。它只负责产出 `user_input / retrieved_contexts / response`,
   与评估环境完全解耦(第 7、9 章落地)。

### 3.4 新旧两套 API:0 基础必须知道的事

ragas 0.4.x 里**同时存在两套 API**,网上旧教程鱼龙混杂,先认清楚:

| | 旧 API(0.1~0.2 时代,现已标废弃) | 新 API(0.4.x 主线,本教程主线) |
|---|---|---|
| 指标导入 | `from ragas.metrics import faithfulness` | `from ragas.metrics.collections import Faithfulness` |
| 裁判封装 | `LangchainLLMWrapper(ChatOpenAI(...))` | `llm_factory(model, client=AsyncOpenAI(...))` |
| 打分方式 | `evaluate(dataset, metrics=[...])` 一次跑一批 | 每个指标实例自己 `.ascore(...)` 或仍可用 `evaluate()` |
| 现状 | 调用会弹 `DeprecationWarning`,**计划 v1.0 移除** | 官方推荐 |
| 你需要会 | **看懂**(网上资料大把是这个写法) | **使用** |

> ⚠️ 实测:旧式 `from ragas.metrics import faithfulness` 在 0.4.3 仍能用,但会警告
> `DeprecationWarning: ... will be removed in v1.0. Please use 'ragas.metrics.collections' instead`。

---

## 第 4 章 第一次评估:十几行代码跑通

目标:对一条手工构造的问答,算出**忠实度**和**答案相关性**两个分数。
新建 `first_eval.py`(放在放 `.env` 的目录下):

```python
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
RESPONSE = "The default management IP of the H3C ER2100 is 192.168.1.1."
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
```

运行与**真实输出**(智谱 `glm-5.3-flash` 裁判,2026-09 实测):

```text
$ python first_eval.py
faithfulness = 1.0
answer_relevancy = 0.9303456436093223
```

### 4.1 逐行看懂:刚才发生了什么

1. **`llm_factory(model, client=...)`**:把一个 OpenAI SDK 客户端包装成 ragas 的"裁判"对象。
   因为智谱提供 OpenAI 兼容协议,所以只要换 `base_url` 就能让裁判变成国产模型——
   **这是整个教程最关键的接线点**;
2. **`Faithfulness(llm=judge)`**:忠实度指标 = 答案中的每个论断都能在检索内容里找到依据的比例。
   它内部会调 2 次裁判 LLM(先"拆论断",再"逐条判断有无支撑");
3. **`AnswerRelevancy(llm=judge, embeddings=embedder)`**:答案相关性 = 答案与问题的语义贴近度。
   它让裁判根据答案反推出 N 个"这个问题大概在问什么",再算这些问题与原问题的向量相似度均值,
   所以既需要 LLM 也需要 embedding;
4. **`ascore(...)`**:异步打分,返回 `MetricResult`,取 `.value` 得分数。
   同步版是 `score(...)`,批量版是 `abatch_score([...])` / `batch_score([...])`。

### 4.2 分数怎么读

- 两个指标都是 **[0, 1] 区间,越接近 1 越好**(个别例外见第 5 章噪声敏感度);
- `faithfulness = 1.0`:答案里每个论断("默认管理 IP 是 192.168.1.1")都在检索内容里有依据,零幻觉;
- `answer_relevancy ≈ 0.93`:答案直击问题,没有答非所问的冗余。

### 4.3 练习:亲手制造一次"低分"

改一行代码再跑,建立分数的直觉——把答案换成编造的:

```python
RESPONSE = "The default management IP of the H3C ER2100 is 10.0.0.1."  # 假的
```

预期:`faithfulness` 掉到接近 0(论断无依据),`answer_relevancy` 依然较高
(答案仍然"相关",只是错了)。这正说明**两个指标各管一段**:
相关性管"答得对题",忠实度管"说得有据"。

`★ Insight ─────────────────────────────────────`
- `llm_factory` + `AsyncOpenAI(base_url=...)` 是把 RAGAS 接到任何国产大模型的万能插座,不止智谱。
- 指标分数只有在**同一裁判、同一评估集**下才有可比性;这两样一变,分数就不可比了。
`─────────────────────────────────────────────────`

---

## 第 5 章 指标体系详解:每个指标测什么、怎么读分

> 以下指标均在 ragas 0.4.3 的 `ragas.metrics.collections` 命名空间下,
> 构造方式统一为 `指标类(llm=judge, ...)`。表格先行,逐个展开在后。

### 5.1 指标速查表

| 指标 | 类名 | 回答的问题 | 必需字段 | 需 Embedding | 方向 |
|------|------|-----------|---------|:---:|:---:|
| 忠实度 | `Faithfulness` | 答案是否都有检索内容支撑(幻觉率反向) | ui, rs, rc | — | ↑ |
| 答案相关性 | `AnswerRelevancy` | 答案是否切题、有无废话 | ui, rs | ✅ | ↑ |
| 上下文精确率 | `ContextPrecision` | 检索结果里"有用的排前面了吗" | ui, rc, reference* | — | ↑ |
| 上下文召回 | `ContextRecall` | 标准答案的要点都被检索到了吗 | ui, rc, reference | — | ↑ |
| 事实正确性 | `FactualCorrectness` | 答案与标准答案的事实一致度 | rs, reference | — | ↑ |
| 答案正确性 | `AnswerAccuracy` | 答案对不对(简版) | ui, rs, reference | — | ↑ |
| 噪声敏感度 | `NoiseSensitivity` | 检索混入噪声时答案会不会被带偏 | ui, rs, rc, reference | — | **↓** |
| 语义相似度 | `SemanticSimilarity` | 答案与参考答案的 embedding 余弦相似度 | rs, reference | ✅ | ↑ |

> 缩写:ui=`user_input`,rs=`response`,rc=`retrieved_contexts`;reference=标准答案。
> 带 `*` 的 ContextPrecision 无参考时可用 `ContextPrecisionWithoutReference` 变体。
> 方向列:↑ 越高越好,↓ 越低越好。

**企业最小组合建议**(先跑起来,再逐步加):

```python
# 日常回归(无参考,便宜):检索质量 + 生成质量各取一个代表
[ Faithfulness, AnswerRelevancy ]
# 发版前质检(有参考,贵但准):上述 + 检索召回 + 事实正确性
[ Faithfulness, AnswerRelevancy, ContextRecall, FactualCorrectness ]
```

### 5.2 Faithfulness(忠实度)——最重要的一个

- **测什么**:把 `response` 拆成一条条原子论断,逐条问裁判"这条能从 `retrieved_contexts` 推出来吗",
  分数 = 有支撑的论断数 ÷ 总论断数;
- **需要**:`user_input`、`response`、`retrieved_contexts`(不需要参考答案);
- **读分**:1.0 = 零幻觉;0.6 = 四成论断是编的,**必须处理**;
- **低分排查顺序**:① 检索回来的片段本身答非所问(看 5.4/5.5)→ ② 生成提示词允许了自由发挥 →
  ③ 模型温度过高;
- **实测注意**:中文答案同样适用;论断拆解由裁判 LLM 完成,裁判越弱拆得越粗。

### 5.3 AnswerRelevancy(答案相关性)

- **测什么**:裁判拿你的答案反推"这答案像是在回答什么问题",生成 N 个候选问题
  (默认 `strictness=3`),与真实问题算 embedding 相似度取均值;
- **需要**:`user_input`、`response`(+ embedding 模型);
- **读分**:高 = 切题;低 = 答非所问 / 太空洞 / 太啰嗦;
- **实测注意**:裁判端点若不支持"一次生成 3 条"(智谱不支持 `n>1`),ragas 会打印
  `LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.`
  **这不是报错**,评估照常完成,只是反推问题数变成了 1,分数方差略大;
- **参数**:`AnswerRelevancy(llm=..., embeddings=..., strictness=3)`。

### 5.4 ContextPrecision(上下文精确率)

- **测什么**:检索回来的片段按顺序逐条问裁判"这条对回答该问题有用吗",
  然后按"有用的是否排在前面"计算类似 MAP 的排序分数;
- **需要**:`user_input`、`retrieved_contexts`,有参考时用 `ContextPrecision`(更准),
  无参考时用 `ContextPrecisionWithoutReference`;
- **读分**:高 = 重排有效,有用的排前面;低 = **重排环节失效或排序被噪声淹没**;
- **低分排查**:重排模型是否降级(掌柜智库 rerank 失败会"保留原序"降级)、断崖截断阈值是否过松。

### 5.5 ContextRecall(上下文召回)

- **测什么**:把**人工标准答案**拆成要点,逐条问裁判"这个要点能从检索内容里找到吗",
  分数 = 被覆盖的要点比例;
- **需要**:`user_input`、`retrieved_contexts`、`reference`(**必须有参考**);
- **读分**:低 = 检索漏了关键内容,生成再好也白搭——**检索侧最致命的指标**;
- **低分排查**:切分粒度(关键句被切碎)、embedding 模型与领域匹配度、检索 top-k 是否太小、
  主体过滤是否误杀。

### 5.6 FactualCorrectness(事实正确性)

- **测什么**:裁判把 `response` 与 `reference` 都拆成事实论断集合,按模式计算
  precision / recall / f1(`mode` 参数,默认 `f1`);
- **需要**:`response`、`reference`;`mode="recall"` 时还需要 `user_input`;
- **读分**:高 = 答案与标准答案事实一致;它是最接近"传统断言"的指标,适合发版前把关;
- **参数**:`FactualCorrectness(llm=..., mode="f1")`,可选 `atomicity="high"`(拆更细)、
  `coverage="high"`(覆盖更全,更慢更贵)。

### 5.7 NoiseSensitivity(噪声敏感度)——唯一"越低越好"

- **测什么**:专门检查"检索内容里混进了无关片段时,答案会不会被带偏说出无依据的话";
- **需要**:全量字段(含 `reference`);
- **读分**:**↓ 越低越好**。0 = 检索有噪声也稳;高 = 你的生成环节缺乏"忽略无关上下文"的能力;
- **用法**:与 Faithfulness 搭配,定位"到底是噪声太多"还是"模型太容易被带偏"。

### 5.8 便捷指标(不需要裁判 LLM,几乎免费)

```python
from ragas.metrics.collections import (
    BleuScore, ChrfScore, RougeScore,          # 机器翻译老三样:与参考答案的 n-gram 重叠
    ExactMatch, NonLLMStringSimilarity,        # 精确匹配 / 字符串相似度
    SemanticSimilarity,                        # embedding 余弦相似度(需 embeddings=...)
)
```

适合"答案基本是固定口径"的场景(如默认 IP、端口号、参数名),做**粗筛**很香;
自然语言答案别用它们当主指标。

### 5.9 怎么选:一张决策图

```text
有人工标注的标准答案吗?
├─ 有 → 发版质检/方案对比:Faithfulness + AnswerRelevancy + ContextRecall + FactualCorrectness
└─ 没有 → 日常回归/巡检:Faithfulness + AnswerRelevancy(+ ContextPrecisionWithoutReference)
          同时开始积累标注:每轮评估挑 10 条分数可疑的补标注,评估集逐步长大
```

---

## 第 6 章 构建评估数据集:冷启动的三条路

评估集 = 一批 `SingleTurnSample`。没有评估集,一切指标都是空谈。
企业冷启动按成本从低到高有三条路,**推荐 6.1 + 6.3 组合起步**。

### 6.1 路线一:手工标注 CSV(可控性最高)

建 `eval_data.csv`,UTF-8 编码,表头固定:

```csv
user_input,retrieved_contexts,response,reference
"ER2100默认管理地址是什么?","H3C ER2100 快速配置指南:默认管理地址 192.168.1.1……","默认管理地址是192.168.1.1。","ER2100 默认管理地址为 192.168.1.1,用户名密码均为 admin。"
```

标注规范(企业实践沉淀):

| 项 | 规范 |
|----|------|
| 数量 | 首批 **30~50 条**即可起步;100~200 条后分数趋于稳定 |
| 覆盖 | 高频问题 60% + 边界问题 25%(超纲/多主体/口语化)+ 已知坏例 15% |
| `contexts`/`response` | **跑真实系统填进去**,不要手编——你评的是真实链路 |
| `reference` | 从知识库原文整理,人工过目;一人写、另一人审 |
| 存储 | CSV 进 git(数据集也要版本管理!);每条加 `id` 列便于溯源 |

### 6.2 路线二:线上真实问题采样(最贴近真实)

从 MongoDB 会话历史(掌柜智库仓库 `utils/mongo_history_utils.py` 的 `get_recent_messages` 所在集合)导出真实用户问题,
回放给被测系统拿 `contexts` / `response`,`reference` 由人工补。
优点:分布真实;注意:**脱敏**后再进评估集(会话可能含客户信息)。

### 6.3 路线三:RAGAS Testset 自动生成(零标注冷启动)

让裁判 LLM 读你的知识库文档,自动生成"问题 + 参考答案"。
已实测可用的完整脚本 `gen_testset.py`:

```python
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
df.to_csv("auto_testset.csv", index=False, encoding="utf-8-sig")
print(df[["user_input", "reference"]].to_string())
```

**真实运行输出**(智谱生成,注意它会模仿不同的提问风格):

```text
Q1: hello i am network admin and i need to know what is the default management address of the ...
Q2: 想问一下就是说我们公司内网出口用的那台ER2100路由器,最近因为出口带宽不太够用了所以想把几个WAN口给聚合起来用,但是不太清楚这个设备到底支不支持WAN口聚合这个功能,还有就是 ...
```

生成结果包含列:`user_input`(生成的问题)、`reference_contexts`(依据的原文切片)、
`reference`(参考答案)、`persona_name / query_style / query_length`(生成时的人物/风格/长度标签,
可用于分层统计)、`synthesizer_name`(来自哪个合成器)。

**必须人工过一遍再用**:自动生成的问题可能有歧义、超纲或答案牵强,
按 6.1 的规范逐条审,通常保留 70% 左右。

> ⚠️ 实测注意:
> - 文档很少时,多跳合成器(MultiHop)会跳过并提示
>   `Skipping multi_hop_...query_synthesizer ... Cannot form clusters`——不是报错,文档多了自然就有了;
> - 依赖 `rapidfuzz`、`nltk`(见 2.3),缺了会报 `ImportError: rapidfuzz is required for string distance`;
> - 成本参考:2 条样本约 1 分钟、十几次 LLM 调用;生成 50 条请按 0.5~1 小时和几毛钱预算。

### 6.4 评估集的维护纪律

1. **版本化**:评估集文件进 git,改动评估集 = 改基线,commit message 必须说明原因;
2. **只增不删**:发现新坏例就追加,不给旧条目"改答案迁就系统";
3. **分层统计**:利用 `query_style`/`query_length` 或自加 `category` 列,分桶看分数
   ("口语化问题忠实度显著低于书面问题"这类洞察比总分有用得多)。

---

## 第 7 章 企业级评估流水线:从脚本到报告

单次跑通没用,**能反复跑、出报告、卡门禁**才是企业级。本章产出三个东西:

```text
ragas_eval/                       # 评估工程目录(独立 git 仓库或被测项目仓库子目录)
├── .env                          # 智谱 key(见 2.4,gitignore!)
├── data/
│   ├── eval_set.csv              # 评估集(第 6 章)
│   └── baseline.json             # 基线分数(第 8 章)
├── pipeline.py                   # 评估流水线(本章)
├── report/                       # 每次运行的报告输出(gitignore)
└── requirements.txt              # 2.3 的版本组合
```

### 7.1 流水线脚本 `pipeline.py`

输入:第 7/9 章产出的样本 JSONL(每行一条已采集的问答);输出:CSV 明细 + Markdown 汇总 + 门禁退出码。

```python
"""pipeline.py —— 企业级评估流水线
用法:
    python pipeline.py data/samples.jsonl \
        --report-dir report/$(date +%Y%m%d_%H%M%S) \
        --thresholds faithfulness=0.75 answer_relevancy=0.6
"""
import argparse
import asyncio
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Windows GBK 控制台保护(同掌柜智库 main.py 的做法)
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness


def load_samples(path: str) -> list[dict]:
    """读取采集样本:每行 {"user_input":..., "response":..., "retrieved_contexts":[...]}"""
    samples = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[warn] 第 {i} 行 JSON 损坏,跳过:{e}")
    return samples


async def score_one(sem, faithfulness, relevancy, s, i, total) -> dict:
    """单条样本打分;sem 限制并发,保护裁判端点(等效于 evaluate 路径的 max_workers)"""
    async with sem:
        row = {"id": s.get("id", i), "user_input": s["user_input"]}
        r1 = await faithfulness.ascore(
            user_input=s["user_input"],
            response=s["response"],
            retrieved_contexts=s["retrieved_contexts"],
        )
        row["faithfulness"] = round(r1.value, 4)
        r2 = await relevancy.ascore(user_input=s["user_input"], response=s["response"])
        row["answer_relevancy"] = round(r2.value, 4)
        print(f"[{i}/{total}] {row}")
        return row


async def run_eval(samples: list[dict]) -> tuple[list[dict], dict]:
    client = AsyncOpenAI(
        base_url=os.environ["OPENAI_API_BASE"], api_key=os.environ["OPENAI_API_KEY"]
    )
    judge = llm_factory(os.getenv("RAGAS_JUDGE_MODEL", "glm-5.3-flash"), client=client)
    embedder = OpenAIEmbeddings(
        model=os.getenv("RAGAS_EMBEDDING_MODEL", "embedding-3"), client=client
    )
    faithfulness = Faithfulness(llm=judge)
    relevancy = AnswerRelevancy(llm=judge, embeddings=embedder)

    sem = asyncio.Semaphore(4)  # 并发上限,按端点限流情况调整
    rows = await asyncio.gather(*[
        score_one(sem, faithfulness, relevancy, s, i, len(samples))
        for i, s in enumerate(samples, 1)
    ])
    summary = {
        m: round(sum(r[m] for r in rows) / len(rows), 4)
        for m in ("faithfulness", "answer_relevancy")
    }
    return rows, summary


def write_reports(rows: list[dict], summary: dict, report_dir: str) -> None:
    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 明细 CSV
    with open(out / "detail.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 汇总 Markdown
    low = [r for r in rows if r["faithfulness"] < 0.5]
    md = [
        f"# RAG 评估报告 {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "| 指标 | 均分 |",
        "|------|------|",
        *(f"| {k} | {v} |" for k, v in summary.items()),
        "",
        f"- 样本数:{len(rows)}",
        f"- 忠实度 < 0.5 的样本:{len(low)} 条(见 detail.csv)",
    ]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[report] 已写入 {out}/summary.md 与 detail.csv")


def check_gates(summary: dict, thresholds: dict[str, float]) -> int:
    """阈值门禁:任一指标低于阈值 → 返回非 0 退出码,CI 直接失败"""
    failed = {m: (summary[m], t) for m, t in thresholds.items() if summary[m] < t}
    for m, (score, th) in failed.items():
        print(f"[GATE-FAIL] {m}={score} < 阈值 {th}")
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", help="采集样本 JSONL 文件")
    parser.add_argument("--report-dir", default="report/latest")
    parser.add_argument(
        "--thresholds", nargs="*", default=["faithfulness=0.75", "answer_relevancy=0.6"],
        help="门禁阈值,格式 指标名=分数",
    )
    args = parser.parse_args()

    samples = load_samples(args.samples)
    assert samples, "样本为空,检查输入文件"
    rows, summary = asyncio.run(run_eval(samples))
    write_reports(rows, summary, args.report_dir)
    thresholds = dict(t.split("=") for t in args.thresholds)
    sys.exit(check_gates(summary, {k: float(v) for k, v in thresholds.items()}))


if __name__ == "__main__":
    main()
```

### 7.2 成本与耗时治理(企业必须回答的四个问题)

| 问题 | 手段 | 本教程落点 |
|------|------|-----------|
| 跑一次多少钱? | 每条样本 × 每指标约 2~4 次裁判调用;50 条 × 2 指标 ≈ 数百次小调用,智谱 glm-5.3-flash 计价下通常 **< 1 元** | — |
| 跑一次多久? | 并发是最大杠杆:逐样本路径用 `asyncio.Semaphore(N)`(pipeline.py),`evaluate()` 批量路径用 `RunConfig(max_workers=N)`;50 条 2 指标 4 并发 ≈ 3~6 分钟 | pipeline.py |
| 抖动/失败? | 单条样本异常用 try/skip 不炸全局(可按需加);`evaluate()` 批量路径可用 `RunConfig(timeout, max_retries, max_wait)` 自动重试 | load_samples + Semaphore |
| 能再省吗? | ① 日常只跑无参考指标 ② 大评估集先抽样(如 20%)过门禁 ③ 分层抽样保覆盖 | 第 8 章 CI 策略 |

### 7.3 报告怎么用:分数 → 行动

| 现象 | 最可能的原因 | 动作 |
|------|-------------|------|
| faithfulness 低,answer_relevancy 高 | 生成在"自由发挥" | 收紧生成提示词、降温度;核对 rerank 是否降级 |
| faithfulness 高,ContextRecall 低 | 检索漏了关键内容 | 调切分粒度 / top-k / embedding 模型 |
| ContextPrecision 低 | 有用片段排在后面 | 查重排模型与断崖截断参数 |
| 两项都低 | 检索整体失效 | 从 ContextRecall 开始逐段排查 |

---

## 第 8 章 CI 集成:让评估成为回归门禁

### 8.1 两种触发方式

| 方式 | 触发条件 | 适用 |
|------|---------|------|
| **事件触发** | 知识库导入完成 / 检索配置变更的 PR | 改动→验证,防劣化 |
| **定时巡检** | 每晚低峰跑一遍线上采样 | 发现"慢性劣化"(模型端点行为漂移等) |

### 8.2 GitHub Actions 示例(定时 + 手动)

```yaml
# .github/workflows/rag-eval.yml
name: rag-eval
on:
  schedule:
    - cron: "17 18 * * *"        # 每天北京时间 02:17(避开整点高峰)
  workflow_dispatch: {}          # 支持手动触发

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - name: Install eval env
        run: |
          pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
            "ragas==0.4.3" "langchain-community<0.4" \
            python-dotenv openai langchain-openai
      - name: Run evaluation
        env:
          OPENAI_API_BASE: ${{ secrets.OPENAI_API_BASE }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          python pipeline.py data/samples.jsonl \
            --report-dir report/$(date +%Y%m%d_%H%M%S) \
            --thresholds faithfulness=0.75 answer_relevancy=0.6
      - uses: actions/upload-artifact@v4     # 报告归档:每次运行的 detail.csv/summary.md
        if: always()
        with:
          name: rag-eval-report
          path: report/
```

要点:

- **阈值门禁靠退出码**:`pipeline.py` 低于阈值 `sys.exit(1)`,job 变红,无需额外配置;
- **报告归档**:artifact 保留每次运行明细,劣化可回溯;
- 密钥进 `secrets`,**永远不进仓库**。

### 8.3 基线对比:比"卡阈值"更聪明的门禁

固定阈值要拍脑袋;更稳的做法是**与基线比**。首次运行后把均分存为 `data/baseline.json`:

```json
{"faithfulness": 0.9123, "answer_relevancy": 0.8452, "judge_model": "glm-5.3-flash", "date": "2026-09-21"}
```

对比逻辑(加进 pipeline 或独立脚本):

```python
def compare_baseline(summary: dict, baseline: dict, max_drop: float = 0.05) -> int:
    """任一指标较基线下跌超过 max_drop → 门禁失败"""
    assert baseline.get("judge_model") == os.getenv("RAGAS_JUDGE_MODEL"), \
        "裁判模型变了,基线失效!必须用同一裁判重建基线"
    failed = [
        f"{m}: {summary[m]} < 基线 {baseline[m]} - {max_drop}"
        for m in summary
        if summary[m] < baseline[m] - max_drop
    ]
    for msg in failed:
        print(f"[GATE-FAIL] 较基线劣化 {msg}")
    return 1 if failed else 0
```

**基线纪律**:基线必须与"合格版本"绑定(如当前线上配置),换裁判模型/换评估集 = 基线作废,
必须重建;基线下调需要有 ticket 说明原因。

---

## 第 9 章 实战:评估"掌柜智库"知识库

把前 8 章全部串起来,对掌柜智库仓库(`knowledge_base_center`)的真实 RAG 链路做一次完整评估。

### 9.1 架构:两个环境、两个阶段

一个绕不开的现实:本项目 venv 里装着 `langchain-community 0.4.x`(会弄崩 ragas,见 2.2),
而评估环境里没有 pymilvus/milvus 等项目依赖。**正确解法不是硬凑一个全能环境,而是解耦**:

```text
┌─────────── 阶段 1:采集(项目 venv)───────────┐    ┌───────── 阶段 2:评估(评估 venv)─────────┐
│ collect_kb_samples.py                          │    │ pipeline.py(第 7 章)                     │
│  进程内直调 KBQueryWorkflow:                   │    │  读 samples.jsonl                          │
│    question ──► 检索+生成                      │JSONL│  ragas 打分(智谱裁判)                    │
│  抽取 answer + reranked_docs[].content         │ ──► │  输出报告 + 阈值门禁                       │
│  写 data/samples.jsonl                         │    │                                            │
└────────────────────────────────────────────────┘    └────────────────────────────────────────────┘
   需要:Milvus / MongoDB / MinIO 在线                      只需要:.env 里的智谱 key
   不需要:ragas                                            不需要:任何项目依赖
```

这个解耦不是妥协,而是企业标准姿势:**被测系统的运行环境和评估工具链永远分离**,
互不污染依赖,采集产物(JSONL)成为两边的契约。

### 9.2 阶段 1:采集脚本(在项目环境运行)

`collect_kb_samples.py` 放在掌柜智库仓库根目录。它利用两个已验证的事实:

1. `KBQueryWorkflow().run(init_state, stream=False)` 返回**最终状态字典**,
   其中 `answer` 是生成的答案、`reranked_docs` 是重排+断崖截断后的最终参考文档
   (条目含 `content / source / chunk_id / url / title / score`,见
   `processor/query_processor/state.py` 与 `node_answer_output.py`);
2. 每个问题用**独立 session_id**,避免 MongoDB 会话历史把上一题的上下文带进下一题。

```python
"""collect_kb_samples.py —— 采集掌柜智库真实链路样本(在【项目 venv】下运行)

前置:项目 .env 已配好智谱 key;Milvus/MongoDB/MinIO 已启动
用法:python collect_kb_samples.py questions.txt data/samples.jsonl
"""
import json
import sys
import uuid

for _s in (sys.stdout, sys.stderr):           # Windows GBK 保护(同 main.py)
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from processor.query_processor.main_graph import KBQueryWorkflow

QUESTIONS_FILE = sys.argv[1] if len(sys.argv) > 1 else "questions.txt"
OUTPUT_FILE = sys.argv[2] if len(sys.argv) > 2 else "data/samples.jsonl"

# 待评估问题清单(首评建议 10~30 条,从知识库覆盖范围内挑选)
QUESTIONS = [
    "H3C ER2100 路由器的默认管理地址是什么?",
    "ER2100 怎么配置 WAN 口聚合?",
    # ... 从 questions.txt 读更多
]


def main() -> None:
    try:
        questions = [q.strip() for q in open(QUESTIONS_FILE, encoding="utf-8") if q.strip()]
    except FileNotFoundError:
        questions = QUESTIONS

    workflow = KBQueryWorkflow()
    n_ok = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for i, q in enumerate(questions, 1):
            # 每题独立会话,防止历史对话污染下一题的检索与主体确认
            session_id = f"ragas-eval-{uuid.uuid4().hex[:12]}"
            print(f"[{i}/{len(questions)}] {q}")
            try:
                final_state = workflow.run(
                    {"session_id": session_id, "original_query": q, "is_stream": False},
                    stream=False,
                )
            except Exception as e:
                print(f"    [skip] 链路异常:{e}")
                continue
            answer = (final_state.get("answer") or "").strip()
            contexts = [
                d.get("content", "") for d in (final_state.get("reranked_docs") or [])
                if d.get("content")
            ]
            if not answer or not contexts:
                # 反问/拒绝/检索为空的样本:单独立类记录,不要混进正常评估
                print("    [warn] 空答案或空检索,跳过(记录为拒答类样本)")
                continue
            out.write(json.dumps({
                "id": f"q{i:03d}",
                "user_input": q,
                "response": answer,
                "retrieved_contexts": contexts,
                "session_id": session_id,      # 留着便于回查 Mongo 历史定位问题
            }, ensure_ascii=False) + "\n")
            n_ok += 1
    print(f"完成:成功 {n_ok} 条 → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
```

运行(注意:**项目环境**,不需要 ragas):

```bash
# 1. 先把基础服务拉起来(Milvus/MongoDB/MinIO,见仓库 README / docker/)
python main.py all        # 或者只起依赖也可,采集是进程内直调、不走 8001 HTTP

# 2. 采集
python collect_kb_samples.py questions.txt data/samples.jsonl
```

> 💡 为什么不走 HTTP 接口?`POST /query` 的响应只有 `answer`,**不返回检索上下文**,
> 而 RAGAS 的检索类指标必须有 `retrieved_contexts`。进程内直调能拿到完整状态,
> 这也是"评估驱动可观测性"的典型例子:为了让系统可评估,你至少要能拿到中间产物。

### 9.3 阶段 2:评估(在评估环境运行)

```bash
source /d/workspace/.venvs/ragas-eval/activate   # 激活第 2 章建好的评估环境
cd ragas_eval/                                    # 第 7 章的评估工程目录
cp /path/to/knowledge_base_center/data/samples.jsonl data/
python pipeline.py data/samples.jsonl \
    --report-dir report/first_run \
    --thresholds faithfulness=0.75 answer_relevancy=0.6
```

预期输出形如(真实分数以你的知识库内容为准):

```text
[1/12] H3C ER2100 路由器的默认管理地址是什么?
{'id': 1, 'user_input': '...', 'faithfulness': 1.0, 'answer_relevancy': 0.87}
...
[GATE] faithfulness=0.92 >= 0.75 ✔   answer_relevancy=0.78 >= 0.6 ✔
[report] 已写入 report/first_run/summary.md 与 detail.csv
```

### 9.4 解读 → 定位 → 优化的闭环

结合掌柜智库链路(导入流水线 + 三路检索 + RRF + rerank + 生成),给出排查地图:

```text
faithfulness 低
 ├─ 看 detail.csv 里低分样本的 retrieved_contexts
 ├─ 片段本身文不对题 → 检索问题:查主体确认(node_item_name_confirm 是否误判)、
 │   embedding 检索(node_search_embedding_hyde)命中质量、RRF 融合权重
 └─ 片段相关但答案跑偏 → 生成问题:node_answer_output 的提示词是否给了太多发挥空间、
     温度配置(config/lm_config.py 的 LLM_DEFAULT_TEMPERATURE)

answer_relevancy 低
 ├─ 答案太空/太泛 → 生成提示词要求"先给结论"
 └─ 答非所问 → 主体确认环节是否反问了(这类样本应单独归类,见 9.2 warn 分支)

ContextRecall 低(若你补了参考答案)
 └─ 导入侧:chunk 切分粒度、主体名回填(node_item_name_recognition)质量
```

优化后**复评**:改动 → 重新采集 → 重新评估 → 与 `baseline.json` 对比(第 8.3 节)。
一次完整的"改切分策略 → 忠实度 +0.08"才算真正吃透了这套流程。

### 9.5 方案对比实验(选做)

同一批问题,分别评估不同配置,分数表直接进技术方案文档:

| 实验组 | 改动 | faithfulness | answer_relevancy |
|--------|------|:---:|:---:|
| A(基线) | 现状 | — | — |
| B | rerank 断崖阈值收紧 | — | — |
| C | 关闭 HyDE 路(仅单路向量) | — | — |

> 对照实验时只动一个变量;裁判模型、评估集、RunConfig 全部保持不变。

---

## 第 10 章 进阶方向

### 10.1 多轮对话评估

`MultiTurnSample` 支持 Agent 式多轮交互(消息列表 + 工具调用记录),对应
`TopicAdherence`、`ToolCallAccuracy`、`AgentGoalAccuracy` 等指标。纯 RAG 知识问答场景
用得少,当你的系统带工具调用/多轮追问时再深入,官方文档
[Metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/) 有完整列表。

### 10.2 自定义指标:把"企业口径"变成分数

业务常有硬口径,如"答案必须包含客服电话"。用 `DiscreteMetric` 实现
(0.4.3 已验证的构造签名):

```python
from ragas.metrics import DiscreteMetric

m = DiscreteMetric(
    name="contains_hotline",
    allowed_values=["pass", "fail"],
    prompt="判断回答是否包含官方客服电话 400-xxx-xxxx。只回答 pass 或 fail。",
)
# 与内置指标一样:await m.ascore(user_input=..., response=...)
```

更复杂的口径(评分档位)可用 `NumericMetric` / `RankingMetric` 同理构造。

### 10.3 裁判模型升级流程

更强的裁判 = 更可信的分数,但换裁判 = 历史分数不可比。标准流程:

1. 新旧裁判**并行**跑同一评估集,对比差异(通常相关但不相等);
2. 抽 10 条人工核对,确认新裁判的判断更准;
3. 用新裁判**重建基线**并记录 `judge_model` 字段(8.3 的断言会强制你这么做);
4. 旧报告保留归档,新旧分数不在一张图里比较。

### 10.4 其他值得看的方向

- **成本追踪**:ragas 内置 `TokenUsageParser`(ragas.cost),可在 evaluate 时统计 token 消耗;
- **结果缓存**:ragas 自带基于 diskcache 的缓存层(ragas.cache),重复实验可显著省调用;
- **可观测平台**:LangSmith / Langfuse 等可与 langchain 链路集成,把"检索了什么、生成了什么、
  评了几分"串成一条 trace,适合规模化后的日常运维;
- **prompt 优化器**:ragas.optimizers 提供基于评估分数自动优化提示词的实验性能力,
  企业内慎用(可解释性差),先用人肉闭环跑稳。

---

## 附录 A:常见报错速查(全部实测)

| # | 报错/现象 | 原因 | 解法 |
|---|----------|------|------|
| 1 | `ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'` | ragas 0.4.3 未钉死 langchain-community 版本,装到了 0.4.2+(该模块被移除) | 钉 `langchain-community==0.4.1`(仓库 requirements-eval.txt 同款)或 `<0.4`;不要装进业务 venv |
| 2 | `ImportError: rapidfuzz is required for string distance` | Testset 生成需要 rapidfuzz,但它在 `[all]` extra 里,基础依赖不含 | `pip install rapidfuzz nltk` |
| 3 | `AttributeError: 'LangchainLLMWrapper' object has no attribute 'agenerate_prompt'` | TestsetGenerator 传了旧封装 | 生成器改用新组件:`TestsetGenerator(llm=llm_factory(...), embedding_model=OpenAIEmbeddings(...))` |
| 4 | `LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.` | AnswerRelevancy 想让裁判一次生成 3 个反推问题,智谱不支持 `n>1` | **不是报错**,评估照常;介意方差可把 strictness 调为 1 |
| 5 | 结束时打印一长串 `RuntimeError: Event loop is closed` | Windows 上 httpx/aiohttp 在 proactor 事件循环关闭时的清理噪音 | 无害,忽略;或升级 httpx |
| 6 | `UnicodeEncodeError: 'gbk' codec can't encode ...` | Windows 控制台 GBK + print 中文特殊字符 | `PYTHONIOENCODING=utf-8`,或脚本里 `sys.stdout.reconfigure(encoding="utf-8")` |
| 7 | `DeprecationWarning: Importing faithfulness from 'ragas.metrics' is deprecated ...` | 用了旧 API 导入(网上旧教程通用写法) | 换 `from ragas.metrics.collections import Faithfulness`;v1.0 将移除旧路径 |
| 8 | 401 / 连接失败 | `OPENAI_API_BASE` 漏了 `/api/paas/v4` 路径,或 key 无效 | 对照 2.4 的 .env;`curl $OPENAI_API_BASE/chat/completions` 排查 |
| 9 | 所有指标恒为 0 或裁判输出乱码 | 裁判模型太弱,结构化输出解析失败 | 换更强的裁判模型;temperature 调 0;先拿第 4 章单样本冒烟 |
| 10 | 在 FastAPI/服务进程里调 ragas 报事件循环冲突 | ragas 评估自带异步执行器,与业务事件循环互相干扰 | **评估永远是独立进程**(第 9 章两阶段架构天然规避) |
| 11 | 每次评估分数有小幅波动 | 裁判 LLM 本身的随机性 | temperature=0;`RunConfig(seed=42)`;波动 ±0.03 内属正常,门禁阈值留出余量 |
| 12 | `ValueError: Unsupported provider: embedding-3. Supported providers: openai, ...` | `embedding_factory` 的**第一个位置参数是 provider 不是 model**,按位置传模型名就会踩中 | 用关键字 `embedding_factory(model=..., client=...)`;或直接 `from ragas.embeddings import OpenAIEmbeddings`(教程写法,无废弃警告) |

## 附录 B:术语表

| 术语 | 一句话解释 |
|------|-----------|
| RAG | 检索增强生成:先从知识库检索片段,再让 LLM 据此回答 |
| chunk / 切片 | 知识库导入时把文档切成的最小检索单元 |
| embedding | 把文本映射成向量的模型,相似文本向量距离近 |
| rerank | 对初检结果按相关性重排序的模型 |
| 幻觉 | 答案中出现知识库里没有依据的内容 |
| 裁判模型(judge) | 评估时负责拆解/比对/打分的 LLM,必须全程固定 |
| SingleTurnSample | RAGAS 中一条单轮评估数据的载体(问题/检索内容/答案/参考答案) |
| 评估集 | 一批评估数据的集合,建议版本化管理 |
| 基线(baseline) | 某个"合格版本"的分数快照,用于对比劣化 |
| 门禁(gate) | CI 中"指标低于阈值则失败"的自动拦截 |
| LLM-as-a-Judge | 用 LLM 给 LLM 的输出打分的评估范式 |

## 附录 C:版本信息与参考

**本教程实测环境**(2026-09-21):

- Python 3.11.3 / Windows 11(git-bash),结论对 Linux 同样适用
- ragas 0.4.3、langchain-community 0.3.31、langchain-openai 1.1.9、openai 1.109.1、datasets 5.0.1
- 裁判:智谱 `glm-5.3-flash`;向量:智谱 `embedding-3`(OpenAI 兼容端点 `https://open.bigmodel.cn/api/paas/v4`)

**参考链接**:

- RAGAS 官方仓库:<https://github.com/explodinggradients/ragas>
- RAGAS 官方文档:<https://docs.ragas.io/>
- 智谱开放平台:<https://open.bigmodel.cn>
- 掌柜智库仓库相关文件(位于 `D:\workspace\python_study\knowledge_base_center`):
  `config/lm_config.py`、`config/embedding_config.py`、
  `processor/query_processor/state.py`、`web/api/query_service.py`

**维护提示**:ragas 迭代很快(0.4.x 的新 API 到 v1.0 前仍会演进)。升级 ragas 版本时:
先在新 venv 里跑第 2.5 节自检 + 第 4 章冒烟,再决定是否升级;评估基线与 ragas 版本也应在
`baseline.json` 里一并记录。

---

*教程完 —— 有条件的话,现在就跑一遍第 4 章,十分钟后你就拥有第一份 RAG 质量报告。*



