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


class TestGates(unittest.TestCase):
    def test_higher_better_fails_below_threshold(self):
        self.assertEqual(pipeline.check_gates({"faithfulness": 0.5},
                                              {"faithfulness": 0.75}), 1)

    def test_higher_better_passes_at_threshold(self):
        self.assertEqual(pipeline.check_gates({"faithfulness": 0.75},
                                              {"faithfulness": 0.75}), 0)

    def test_lower_better_fails_above_threshold(self):
        self.assertEqual(pipeline.check_gates({"noise_sensitivity": 0.3},
                                              {"noise_sensitivity": 0.2}), 1)

    def test_lower_better_passes_below_threshold(self):
        self.assertEqual(pipeline.check_gates({"noise_sensitivity": 0.1},
                                              {"noise_sensitivity": 0.2}), 0)

    def test_lower_better_passes_at_threshold(self):
        self.assertEqual(pipeline.check_gates({"noise_sensitivity": 0.2},
                                              {"noise_sensitivity": 0.2}), 0)

    def test_validate_thresholds_rejects_non_enabled_metric(self):
        with self.assertRaises(SystemExit) as cm:
            pipeline.validate_thresholds({"context_recall": 0.8}, {"faithfulness"})
        self.assertEqual(cm.exception.code, 1)


class TestParseThresholds(unittest.TestCase):
    def test_malformed_no_equals_exits(self):
        with self.assertRaises(SystemExit) as cm:
            pipeline.parse_thresholds(["faithfulness075"])
        self.assertEqual(cm.exception.code, 1)

    def test_non_numeric_score_exits(self):
        with self.assertRaises(SystemExit) as cm:
            pipeline.parse_thresholds(["faithfulness=abc"])
        self.assertEqual(cm.exception.code, 1)

    def test_valid_parses(self):
        self.assertEqual(pipeline.parse_thresholds(["faithfulness=0.75"]),
                         {"faithfulness": 0.75})


class TestReport(unittest.TestCase):
    def test_summary_md_has_direction_column(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            rows = [{"id": 1, "user_input": "q",
                     "faithfulness": 0.9, "noise_sensitivity": 0.1}]
            summary = {"faithfulness": 0.9, "noise_sensitivity": 0.1}
            pipeline.write_reports(rows, summary, d, {"faithfulness", "noise_sensitivity"})
            text = (Path(d) / "summary.md").read_text(encoding="utf-8")
            self.assertIn("| 指标 | 方向 | 均分 |", text)
            self.assertIn("| faithfulness | ↑ | 0.9 |", text)
            self.assertIn("| noise_sensitivity | ↓ | 0.1 |", text)


class TestCsvAliases(unittest.TestCase):
    def test_reference_contexts_alias_mapped(self):
        import tempfile
        content = (
            "user_input,reference_contexts,reference\n"
            'q1,"[""片段A"",""片段B""]",r1\n'
        )
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "testset.csv"
            p.write_text(content, encoding="utf-8")
            samples = pipeline.load_samples(str(p))
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["retrieved_contexts"], ["片段A", "片段B"])
        self.assertEqual(samples[0]["reference"], "r1")
        self.assertNotIn("response", samples[0])

    def test_response_column_optional(self):
        import tempfile
        content = 'user_input,response,retrieved_contexts\nq1,a1,"[""c""]"\n'
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "samples.csv"
            p.write_text(content, encoding="utf-8")
            samples = pipeline.load_samples(str(p))
        self.assertEqual(samples[0]["response"], "a1")
        self.assertEqual(samples[0]["retrieved_contexts"], ["c"])

    def test_parse_contexts_single_quoted_list(self):
        # testset 生成器输出 Python 风格单引号列表串
        self.assertEqual(
            pipeline._parse_contexts("['片段A','片段B']"), ["片段A", "片段B"],
        )


class TestValidateSamples(unittest.TestCase):
    def test_missing_reference_exits_for_context_recall(self):
        with self.assertRaises(SystemExit) as cm:
            pipeline.validate_samples(
                [{"user_input": "q", "response": "a", "retrieved_contexts": ["c"]}],
                {"context_recall"},
            )
        self.assertEqual(cm.exception.code, 1)

    def test_missing_response_exits_for_faithfulness(self):
        with self.assertRaises(SystemExit) as cm:
            pipeline.validate_samples(
                [{"user_input": "q", "reference": "r", "retrieved_contexts": ["c"]}],
                {"faithfulness"},
            )
        self.assertEqual(cm.exception.code, 1)

    def test_all_fields_present_passes(self):
        pipeline.validate_samples(
            [{"user_input": "q", "response": "a", "reference": "r",
              "retrieved_contexts": ["c"]}],
            {"noise_sensitivity"},
        )  # 不应抛出/退出


class TestLoaders(unittest.TestCase):
    def test_load_csv_tutorial_format(self):
        import tempfile
        content = (
            "user_input,retrieved_contexts,response,reference,id\n"
            'q1,"[""片段A"",""片段B""]",a1,r1,1\n'
            "q2,,a2,,2\n"
        )
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "samples.csv"
            p.write_text(content, encoding="utf-8")
            samples = pipeline.load_samples(str(p))
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0]["user_input"], "q1")
        self.assertEqual(samples[0]["retrieved_contexts"], ["片段A", "片段B"])
        self.assertEqual(samples[0]["response"], "a1")
        self.assertEqual(samples[0]["reference"], "r1")
        self.assertEqual(samples[0]["id"], "1")
        self.assertNotIn("reference", samples[1])
        self.assertEqual(samples[1]["retrieved_contexts"], [])

    def test_parse_contexts_fallbacks(self):
        self.assertEqual(pipeline._parse_contexts("单一片段"), ["单一片段"])
        self.assertEqual(pipeline._parse_contexts('["a","b"]'), ["a", "b"])
        self.assertEqual(pipeline._parse_contexts(""), [])
        self.assertEqual(pipeline._parse_contexts(["x"]), ["x"])

    def test_load_jsonl_dispatch(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "samples.jsonl"
            p.write_text(json.dumps({"user_input": "q", "response": "a",
                                     "retrieved_contexts": ["c"]}),
                         encoding="utf-8")
            samples = pipeline.load_samples(str(p))
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["retrieved_contexts"], ["c"])


class TestRealDataSmoke(unittest.TestCase):
    """真实数据冒烟:auto_testset.csv(testset 生成格式)加载与字段映射"""

    AUTO = Path(__file__).resolve().parent.parent / "data" / "auto_testset.csv"

    @unittest.skipUnless(AUTO.exists(), "需要 data/auto_testset.csv(gitignore 生成数据)")
    def test_load_auto_testset_csv(self):
        samples = pipeline.load_samples(str(self.AUTO))
        self.assertGreater(len(samples), 0)
        s = samples[0]
        self.assertIn("user_input", s)
        self.assertIn("retrieved_contexts", s)
        self.assertGreater(len(s["retrieved_contexts"]), 0)
        self.assertIn("reference", s)
        self.assertNotIn("response", s)  # 该文件无 response 列


@unittest.skipUnless(
    __import__("os").environ.get("OPENAI_API_BASE") and
    __import__("os").environ.get("OPENAI_API_KEY"),
    "需要 .env 提供 OPENAI_API_BASE / OPENAI_API_KEY",
)
class TestBuildMetricsOffline(unittest.TestCase):
    def test_all_six_metrics_construct_without_network(self):
        import os

        from openai import AsyncOpenAI
        from ragas.embeddings import OpenAIEmbeddings
        from ragas.llms import llm_factory

        client = AsyncOpenAI(base_url=os.environ["OPENAI_API_BASE"],
                             api_key=os.environ["OPENAI_API_KEY"])
        judge = llm_factory(os.getenv("RAGAS_JUDGE_MODEL", "glm-5.3-flash"),
                            client=client)
        embedder = OpenAIEmbeddings(
            model=os.getenv("RAGAS_EMBEDDING_MODEL", "embedding-3"), client=client)
        metrics = pipeline._build_metrics(set(pipeline.ALL_METRICS), judge, embedder)
        self.assertEqual(len(metrics), 6)


if __name__ == "__main__":
    unittest.main()
