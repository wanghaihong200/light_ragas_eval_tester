"""pipeline.py —— RaGas知识库评估流水线
用法:
    python pipeline.py data/samples.jsonl --report-dir report/$(date +%Y%m%d_%H%M%S) --thresholds faithfulness=0.75 answer_relevancy=0.6
    python pipeline.py data/samples.csv --metrics faithfulness answer_relevancy context_precision context_recall factual_correctness noise_sensitivity
"""
import argparse
import ast
import asyncio
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
    NoiseSensitivity,
)

# 指标元数据表:kwargs 即 ragas 0.4.3 ascore 形参(关键字传参,顺序仅作文档);
# higher_better=False 表示越低越好(如 noise_sensitivity)
METRIC_SPECS: dict[str, dict] = {
    "faithfulness":        {"kwargs": ("user_input", "response", "retrieved_contexts"),  "higher_better": True},
    "answer_relevancy":    {"kwargs": ("user_input", "response"),                        "higher_better": True},
    "context_precision":   {"kwargs": ("user_input", "reference", "retrieved_contexts"), "higher_better": True},
    "context_recall":      {"kwargs": ("user_input", "reference", "retrieved_contexts"), "higher_better": True},
    "factual_correctness": {"kwargs": ("response", "reference"),                         "higher_better": True},
    "noise_sensitivity":   {"kwargs": ("user_input", "response", "reference", "retrieved_contexts"), "higher_better": False},
}

ALL_METRICS = tuple(METRIC_SPECS)
METRICS_NEED_REFERENCE = {
    m for m, spec in METRIC_SPECS.items() if "reference" in spec["kwargs"]
}


def _parse_contexts(val) -> list[str]:
    if isinstance(val, list):
        return val
    if isinstance(val, str) and val.strip():
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
        return [val]
    return []


def load_samples(path: str) -> list[dict]:
    """读取采集样本,支持 .jsonl / .csv 格式自动识别"""
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        return _load_csv(path)
    return _load_jsonl(path)


def _load_jsonl(path: str) -> list[dict]:
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


def _load_csv(path: str) -> list[dict]:
    samples = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, raw in enumerate(reader, 1):
            try:
                ctx_raw = raw.get("retrieved_contexts") or raw.get("reference_contexts", "")
                row = {
                    "user_input": raw["user_input"],
                    "retrieved_contexts": _parse_contexts(ctx_raw),
                }
                if raw.get("response"):
                    row["response"] = raw["response"]
                if raw.get("reference"):
                    row["reference"] = raw["reference"]
                if raw.get("id"):
                    row["id"] = raw["id"]
                samples.append(row)
            except KeyError as e:
                print(f"[warn] 第 {i} 行缺少必填列 {e},跳过")
    return samples


def _build_metrics(enabled: set[str], judge, embedder) -> dict[str, object]:
    """按启用列表构造 ragas 指标实例"""
    metrics: dict[str, object] = {}
    if "faithfulness" in enabled:
        metrics["faithfulness"] = Faithfulness(llm=judge)
    if "answer_relevancy" in enabled:
        metrics["answer_relevancy"] = AnswerRelevancy(llm=judge, embeddings=embedder)
    if "context_precision" in enabled:
        metrics["context_precision"] = ContextPrecision(llm=judge)
    if "context_recall" in enabled:
        metrics["context_recall"] = ContextRecall(llm=judge)
    if "factual_correctness" in enabled:
        metrics["factual_correctness"] = FactualCorrectness(llm=judge)
    if "noise_sensitivity" in enabled:
        metrics["noise_sensitivity"] = NoiseSensitivity(llm=judge)
    return metrics


async def _ascore_metric(metric, sample: dict, metric_name: str) -> float:
    """按 METRIC_SPECS 声明的形参组装 ascore 入参并打分"""
    kwargs = {f: sample[f] for f in METRIC_SPECS[metric_name]["kwargs"]}
    result = await metric.ascore(**kwargs)
    return round(result.value, 4)


async def score_one(sem, metrics: dict[str, object], s: dict, i: int, total: int) -> dict:
    """单条样本打分;sem 限制并发,保护裁判端点"""
    async with sem:
        row: dict = {"id": s.get("id", i), "user_input": s["user_input"]}
        for name, metric in metrics.items():
            row[name] = await _ascore_metric(metric, s, name)
        print(f"[{i}/{total}] {row}")
        return row


async def run_eval(
    samples: list[dict], enabled_metrics: set[str],
) -> tuple[list[dict], dict]:
    client = AsyncOpenAI(
        base_url=os.environ["OPENAI_API_BASE"], api_key=os.environ["OPENAI_API_KEY"],
    )
    judge = llm_factory(os.getenv("RAGAS_JUDGE_MODEL", "glm-5.3-flash"), client=client)
    embedder = OpenAIEmbeddings(
        model=os.getenv("RAGAS_EMBEDDING_MODEL", "embedding-3"), client=client,
    )
    metrics = _build_metrics(enabled_metrics, judge, embedder)

    sem = asyncio.Semaphore(4)
    rows = await asyncio.gather(*[
        score_one(sem, metrics, s, i, len(samples))
        for i, s in enumerate(samples, 1)
    ])
    active_names = [m for m in ALL_METRICS if m in enabled_metrics]
    summary = {
        m: round(sum(r[m] for r in rows) / len(rows), 4)
        for m in active_names
    }
    return rows, summary


def write_reports(
    rows: list[dict], summary: dict, report_dir: str, enabled_metrics: set[str],
) -> None:
    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "detail.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    low_faith = (
        [r for r in rows if r.get("faithfulness", 1) < 0.5]
        if "faithfulness" in enabled_metrics else []
    )
    md = [
        f"# RAG 评估报告 {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "| 指标 | 方向 | 均分 |",
        "|------|------|------|",
        *(
            f"| {m} | {'↑' if METRIC_SPECS[m]['higher_better'] else '↓'} | {v} |"
            for m, v in summary.items()
        ),
        "",
        f"- 样本数:{len(rows)}",
        f"- 启用指标:{', '.join(sorted(enabled_metrics))}",
    ]
    if "faithfulness" in enabled_metrics:
        md.append(f"- 忠实度 < 0.5 的样本:{len(low_faith)} 条(见 detail.csv)")
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[report] 已写入 {out}/summary.md 与 detail.csv")


def check_gates(summary: dict, thresholds: dict[str, float]) -> int:
    """阈值门禁:方向感知 — 越高越好低于阈值失败,越低越好高于阈值失败"""
    failed: list[str] = []
    for m, th in thresholds.items():
        score = summary[m]
        hb = METRIC_SPECS[m]["higher_better"]
        if (hb and score < th) or (not hb and score > th):
            failed.append(f"[GATE-FAIL] {m}={score} "
                          f"{'<' if hb else '>'} 阈值 {th}"
                          + ("" if hb else "(该指标越低越好)"))
    for msg in failed:
        print(msg)
    return 1 if failed else 0


def validate_thresholds(thresholds: dict[str, float], enabled: set[str]) -> None:
    """thresholds 引用未启用指标 → 报错退出,防止 CI 假绿"""
    unknown = set(thresholds) - enabled
    if unknown:
        print(f"[error] --thresholds 引用了未启用指标:{sorted(unknown)};"
              f"请将其加入 --metrics 或删除对应阈值")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", help="采集样本文件(.jsonl / .csv)")
    parser.add_argument("--report-dir", default="report/latest")
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=["faithfulness", "answer_relevancy"],
        choices=list(ALL_METRICS),
        help="启用的评估指标(可多选),默认 faithfulness answer_relevancy",
    )
    parser.add_argument(
        "--thresholds", nargs="*", default=["faithfulness=0.75", "answer_relevancy=0.6"],
        help="门禁阈值,格式 指标名=分数",
    )
    args = parser.parse_args()

    enabled = set(args.metrics)

    samples = load_samples(args.samples)
    assert samples, "样本为空,检查输入文件"

    need_ref = enabled & METRICS_NEED_REFERENCE
    if need_ref:
        missing = [i for i, s in enumerate(samples, 1) if "reference" not in s]
        if missing:
            print(f"[error] 指标 {need_ref} 需要 reference 字段,但第 {missing[:5]} 条样本缺失")
            sys.exit(1)

    thresholds = {k: float(v) for k, v in (t.split("=", 1) for t in args.thresholds)}
    validate_thresholds(thresholds, enabled)

    rows, summary = asyncio.run(run_eval(samples, enabled))
    write_reports(rows, summary, args.report_dir, enabled)
    sys.exit(check_gates(summary, thresholds))


if __name__ == "__main__":
    main()