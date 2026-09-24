"""pipeline.py 离线单元测试:零网络调用,零 token 消耗"""
import asyncio
import inspect
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline


class TestMetricSpecs(unittest.TestCase):
    RAGAS_CLASSES = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "AnswerRelevancy",
        "context_precision": "ContextPrecision",
        "context_recall": "ContextRecall",
        "factual_correctness": "FactualCorrectness",
        "noise_sensitivity": "NoiseSensitivity",
    }

    def test_metric_specs_match_ragas_signatures(self):
        from ragas.metrics import collections as rc

        for name, cls_name in self.RAGAS_CLASSES.items():
            cls = getattr(rc, cls_name)
            params = set(inspect.signature(cls.ascore).parameters) - {"self"}
            self.assertEqual(
                set(pipeline.METRIC_SPECS[name]["kwargs"]), params,
                f"{name} 的 METRIC_SPECS kwargs 与 ragas ascore 形参不一致",
            )

    def test_derived_constants(self):
        self.assertEqual(
            pipeline.ALL_METRICS,
            ("faithfulness", "answer_relevancy", "context_precision",
             "context_recall", "factual_correctness", "noise_sensitivity"),
        )
        self.assertEqual(
            pipeline.METRICS_NEED_REFERENCE,
            {"context_precision", "context_recall",
             "factual_correctness", "noise_sensitivity"},
        )


class _FakeMetric:
    """记录 ascore 收到的 kwargs,返回固定分值,不触达 LLM"""

    def __init__(self):
        self.seen = None

    async def ascore(self, **kwargs):
        self.seen = kwargs

        class _R:
            value = 0.5
        return _R()


class TestAscoreKwargs(unittest.TestCase):
    def test_context_precision_gets_reference_not_response(self):
        fake = _FakeMetric()
        sample = {"user_input": "q", "response": "a",
                  "reference": "r", "retrieved_contexts": ["c"]}
        got = asyncio.run(pipeline._ascore_metric(fake, sample, "context_precision"))
        self.assertEqual(got, 0.5)
        self.assertEqual(fake.seen, {"user_input": "q", "reference": "r",
                                     "retrieved_contexts": ["c"]})

    def test_factual_correctness_needs_only_response_and_reference(self):
        fake = _FakeMetric()
        sample = {"user_input": "q", "response": "a", "reference": "r"}
        asyncio.run(pipeline._ascore_metric(fake, sample, "factual_correctness"))
        self.assertEqual(fake.seen, {"response": "a", "reference": "r"})

    def test_noise_sensitivity_gets_all_four_fields(self):
        fake = _FakeMetric()
        sample = {"user_input": "q", "response": "a",
                  "reference": "r", "retrieved_contexts": ["c"]}
        asyncio.run(pipeline._ascore_metric(fake, sample, "noise_sensitivity"))
        self.assertEqual(fake.seen, {"user_input": "q", "response": "a",
                                     "reference": "r", "retrieved_contexts": ["c"]})


if __name__ == "__main__":
    unittest.main()
