"""collect_responses.py —— 模拟 RAG 生成环节,为评估集补上 response 列

TestsetGenerator 产物(auto_testset.csv)只有 问题/参考答案/原文切片,
没有被测系统的生成答案。本项目未接真实 RAG 链路,故用 LLM 基于检索切片
(reference_contexts 视为 retrieved_contexts)生成回答——即"检索固定、
生成真实调用"的模拟被测系统。

用法:
    python collect_responses.py data/auto_testset.csv data/auto_testset_full.csv
"""
import asyncio
import csv
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from openai import AsyncOpenAI

from pipeline import load_samples

PROMPT = (
    "你是一个 RAG 知识问答系统。请仅根据下面的检索内容回答用户问题,"
    "不要编造检索内容之外的信息。用简洁的中文或英文直接回答。\n\n"
    "检索内容:\n{context}\n\n用户问题:{question}\n\n回答:"
)

FIELDS = ["id", "user_input", "retrieved_contexts", "response", "reference"]


async def gen_one(sem, client, model, s: dict, i: int, total: int) -> dict:
    """单条生成;sem 限并发,与 pipeline.py 相同的保护节奏"""
    async with sem:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": PROMPT.format(
                    context="\n".join(s["retrieved_contexts"]),
                    question=s["user_input"],
                ),
            }],
            temperature=0,
        )
        answer = resp.choices[0].message.content.strip()
        print(f"[{i}/{total}] {s['user_input'][:40]}... -> {answer[:50]}...")
        return {
            "id": s.get("id", i),
            "user_input": s["user_input"],
            "retrieved_contexts": s["retrieved_contexts"],
            "response": answer,
            "reference": s["reference"],
        }


async def run(samples: list[dict]) -> list[dict]:
    client = AsyncOpenAI(
        base_url=os.environ["OPENAI_API_BASE"], api_key=os.environ["OPENAI_API_KEY"],
    )
    model = os.getenv("RAGAS_JUDGE_MODEL", "glm-5.3-flash")
    sem = asyncio.Semaphore(4)
    return await asyncio.gather(*[
        gen_one(sem, client, model, s, i, len(samples))
        for i, s in enumerate(samples, 1)
    ])


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    samples = load_samples(src)
    assert samples, "样本为空,检查输入文件"
    rows = asyncio.run(run(samples))
    with open(dst, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in rows:
            out = dict(r)
            out["retrieved_contexts"] = json.dumps(
                r["retrieved_contexts"], ensure_ascii=False)
            writer.writerow(out)
    print(f"[done] {len(rows)} 条 -> {dst}")


if __name__ == "__main__":
    main()
