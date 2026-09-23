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


if __name__ == "__main__":
    unittest.main()
