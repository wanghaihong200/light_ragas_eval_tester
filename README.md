# light_ragas_eval_tester

基于 [RAGAS 0.4.3](https://github.com/explodinggradients/ragas) 的 RAG 系统质量评估流水线(轻量版):
读入评估样本 → 6 大指标 LLM-as-a-Judge 打分 → 生成 Markdown 汇总 + CSV 明细 → 阈值门禁(可作 CI 回归卡点)。

RAG 系统的输出是自然语言,没法用"字符串相等"断言质量。本项目的思路是让裁判 LLM 从
**忠实度、切题度、检索精确/召回、事实正确性、噪声鲁棒性** 六个维度量化打分,
让"这次改动有没有让 RAG 变差"变成一个可比的数字。

## 快速开始

> ⚠️ ragas 0.4.3 对 `langchain-community` 版本敏感,必须用独立虚拟环境
> (装进业务 venv 会导入即崩,详见 `docs/ragas_tutorial.md` 2.2 节踩坑记录)。

```bash
# 1. 建独立环境(Python 3.9 ~ 3.11)
python -m venv .venv
source .venv/Scripts/activate        # git-bash / Windows

# 2. 安装依赖(国内镜像)
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 3. 配置 .env(见下方"配置说明")
cp .env.example .env  # 或手动创建

# 4. 环境自检:通过则打印 4 行 OK
python env_tester.py

# 5. 跑一次评估(仓库自带 2 条样本)
python pipeline.py data/auto_testset_full.csv --report-dir report/first_run
```

## 项目结构

```
light_ragas_eval_tester/
├── pipeline.py             # 评估流水线主程序:加载样本 → 打分 → 报告 → 门禁
├── collect_responses.py    # 采集脚本:模拟 RAG 生成环节,为评估集补上 response 列
├── env_tester.py           # 安装自检:ragas 可导入 + 裁判端点连通性
├── data/
│   ├── auto_testset.csv        # TestsetGenerator 生成的问题集(无 response)
│   └── auto_testset_full.csv   # 补全 response 后的完整评估集
├── report/                 # 每次运行的报告输出(summary.md + detail.csv)
├── tests/
│   └── test_pipeline.py    # 离线单元测试(不调裁判 API)
├── docs/                   # 深度教程(本地文档,未入库)
└── requirements.txt        # 实测版本组合(langchain-community 钉版不能省)
```

## 评估指标

| 指标 | 回答的问题 | 方向 | 必需字段 |
|------|-----------|:---:|---------|
| `faithfulness` | 答案是否都有检索内容支撑(幻觉率反向) | ↑ | user_input, response, retrieved_contexts |
| `answer_relevancy` | 答案是否切题、有无废话 | ↑ | user_input, response |
| `context_precision` | 检索结果里"有用的排前面了吗" | ↑ | user_input, reference, retrieved_contexts |
| `context_recall` | 标准答案的要点都被检索到了吗 | ↑ | user_input, reference, retrieved_contexts |
| `factual_correctness` | 答案与标准答案的事实一致度 | ↑ | response, reference |
| `noise_sensitivity` | 检索混入噪声时答案会不会被带偏 | **↓** | user_input, response, reference, retrieved_contexts |

- ↑ 越高越好,↓ 越低越好(门禁方向感知)
- 无参考指标(`faithfulness` / `answer_relevancy`)便宜,适合日常回归;
  有参考指标适合发版前质检、方案对比
- 启用指标所需的字段缺一不可,流水线启动前会逐条校验、快速失败

## 两阶段评估流程

评估样本 = `user_input / retrieved_contexts / response / reference`。
`response` 必须来自**被测系统**,`reference`(标准答案)来自人工标注或自动生成:

```text
阶段 1:采集                          阶段 2:评估
┌──────────────────────────┐        ┌──────────────────────────────┐
│ 真实 RAG 系统:           │        │ pipeline.py                  │
│  问题 → 检索+生成 → 落盘  │  CSV   │  6 指标打分(裁判 LLM)      │
│ 或 collect_responses.py  │  ───►  │  → report/<dir>/summary.md   │
│ (模拟生成,补 response)  │ JSONL  │  → detail.csv + 阈值门禁     │
└──────────────────────────┘        └──────────────────────────────┘
```

```bash
# 阶段 1:评估集只有问题+参考答案时,模拟 RAG 生成补上 response
python collect_responses.py data/auto_testset.csv data/auto_testset_full.csv

# 阶段 2:全量 6 指标评估
python pipeline.py data/auto_testset_full.csv \
    --report-dir report/$(date +%Y%m%d_%H%M%S) \
    --metrics faithfulness answer_relevancy context_precision context_recall factual_correctness noise_sensitivity
```

样本文件支持 `.jsonl` / `.csv`,`retrieved_contexts` 列接受 JSON 数组、
Python 列表字面量或单字符串(`reference_contexts` 列名亦可)。

## 报告与门禁

每次运行输出两个文件:

| 文件 | 内容 |
|------|------|
| `summary.md` | 指标均分表(含方向)+ 样本数 + 忠实度低分样本统计 |
| `detail.csv` | 逐样本逐指标分数明细(UTF-8-SIG,Excel 直接打开) |

阈值门禁(方向感知:越高越好低于阈值失败,越低越好高于阈值失败):

```bash
# 任一指标破线 → 退出码 1,CI 直接红
python pipeline.py data/auto_testset_full.csv --report-dir report/latest \
    --metrics faithfulness answer_relevancy \
    --thresholds faithfulness=0.75 answer_relevancy=0.6
```

- `--thresholds` 引用未启用指标、格式错误均报错退出(防 CI 假绿)
- 门禁阈值建议留出 ±0.03 余量:裁判 LLM 本身有随机性,小幅波动属正常

## 测试

```bash
python -m unittest tests.test_pipeline -v
```

全部离线运行(打分路径用桩替换,不消耗裁判 API 额度)。

## 配置说明

`.env`(已 gitignore,密钥不进仓库):

| 变量 | 说明 | 示例 |
|------|------|------|
| `OPENAI_API_BASE` | OpenAI 兼容端点(智谱/阿里云百炼等均可) | `https://open.bigmodel.cn/api/paas/v4` |
| `OPENAI_API_KEY` | 对应端点的 API Key | — |
| `RAGAS_JUDGE_MODEL` | 裁判模型(打分 + 模拟生成) | `glm-5.3-flash` / `qwen3.7-plus` |
| `RAGAS_EMBEDDING_MODEL` | 向量模型(answer_relevancy 需要) | `embedding-3` |

> 纪律:换裁判模型 = 历史分数不可比,基线必须用同一裁判重建。

## TODO

- [ ] 多轮对话评测机制待做(`MultiTurnSample` + `TopicAdherence` / `ToolCallAccuracy` / `AgentGoalAccuracy`,适用于带工具调用/多轮追问的 Agent 场景)

## 深入文档

`docs/ragas_tutorial.md` 为本地深度教程(未入库):从零搭建评估环境、8 个指标逐个讲透、
评估集构建三条路、CI 集成与基线对比、常见报错速查(全部实测)。
