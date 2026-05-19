"""
tests/test_rag_ablation.py
===========================

run_rag_ablation.py 单元测试.

测试覆盖:
  - load_ceiling_only_from_jsonl fast path (从 jsonl 自身加载)
  - load_ceiling_only_from_jsonl slow path (调 full_dataset_loader)
  - check_anti_cheat 检测重叠
  - build_prompt_with_rag (含 inject vs 不含)
  - load_existing_results (resume 支持)

运行:
  cd /home/claude && python -m unittest tests.test_rag_ablation -v
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
))

from run_rag_ablation import (
    load_ceiling_only_from_jsonl,
    check_anti_cheat,
    build_prompt_with_rag,
    load_existing_results,
)
from knowledge_tree.core import KnowledgeNode, KnowledgeTree, WorkedExample


# ============================================================================
# load_ceiling_only_from_jsonl
# ============================================================================

class TestLoadCeiling(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _write_jsonl(self, records: list[dict]) -> str:
        path = os.path.join(self.tmpdir, "baseline.jsonl")
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return path

    def test_fast_path_jsonl_with_full_fields(self):
        """jsonl 含完整 question/gt_answer: 直接加载, 不调 full_dataset_loader."""
        records = [
            {"sample_id": 0, "aime_id": "2024-I-1", "question": "Q1",
             "gt_answer": "100", "is_correct": True},
            {"sample_id": 1, "aime_id": "2024-I-2", "question": "Q2",
             "gt_answer": "200", "is_correct": False},  # ceiling
            {"sample_id": 2, "aime_id": "2024-I-3", "question": "Q3",
             "gt_answer": "300", "is_correct": False},  # ceiling
        ]
        path = self._write_jsonl(records)

        # full_dataset_loader 不应被调用
        loader_called = [False]
        def loader():
            loader_called[0] = True
            return []

        samples = load_ceiling_only_from_jsonl(
            path, full_dataset_loader=loader,
        )
        # 应得到 2 道 ceiling 题
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0]["question"], "Q2")
        self.assertEqual(samples[1]["question"], "Q3")
        # loader 没被调用
        self.assertFalse(loader_called[0])

    def test_fast_path_preserves_aime_id(self):
        records = [
            {"sample_id": 0, "aime_id": "2024-I-1", "question": "Q",
             "gt_answer": "1", "is_correct": False},
        ]
        path = self._write_jsonl(records)
        samples = load_ceiling_only_from_jsonl(path)
        self.assertEqual(samples[0]["aime_id"], "2024-I-1")

    def test_slow_path_when_no_question_field(self):
        """jsonl 不含 question: fallback 到 full_dataset_loader."""
        records = [
            {"sample_id": 0, "is_correct": False},  # 无 question
            {"sample_id": 1, "is_correct": True},
        ]
        path = self._write_jsonl(records)

        # Loader 返回 full dataset
        def loader():
            return [
                {"sample_id": 0, "question": "Q0 from loader",
                 "gt_answer": "100", "level": "AIME"},
                {"sample_id": 1, "question": "Q1 from loader",
                 "gt_answer": "200", "level": "AIME"},
            ]

        samples = load_ceiling_only_from_jsonl(
            path, full_dataset_loader=loader,
            aime_id_field="sample_id",
        )
        # 只有 sample_id=0 是 ceiling
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["question"], "Q0 from loader")

    def test_slow_path_no_loader_raises(self):
        """无 question 且 loader=None: 应 raise."""
        records = [
            {"sample_id": 0, "is_correct": False},
        ]
        path = self._write_jsonl(records)
        with self.assertRaises(ValueError):
            load_ceiling_only_from_jsonl(path, full_dataset_loader=None)

    def test_empty_jsonl(self):
        path = self._write_jsonl([])
        samples = load_ceiling_only_from_jsonl(path)
        self.assertEqual(samples, [])

    def test_no_ceiling_returns_empty(self):
        """全 is_correct=True: 返回空."""
        records = [
            {"sample_id": 0, "question": "Q", "gt_answer": "1", "is_correct": True},
        ]
        path = self._write_jsonl(records)
        samples = load_ceiling_only_from_jsonl(path)
        self.assertEqual(samples, [])


# ============================================================================
# check_anti_cheat
# ============================================================================

class TestAntiCheat(unittest.TestCase):

    def _make_tree(self, examples_problems: list[str]) -> KnowledgeTree:
        """构造一个节点, worked_examples 含指定 problem."""
        worked = [
            WorkedExample(
                problem=p, solution_steps=["s"],
                final_answer="a", key_insight="k",
            )
            for p in examples_problems
        ]
        node = KnowledgeNode(
            id="test_concept", title="Test",
            definition="Test concept",
            key_facts=["fact 1", "fact 2"],
            worked_examples=worked,
        )
        return KnowledgeTree(nodes=[node])

    def test_no_overlap(self):
        tree = self._make_tree(["Compute C(5,2) using binomial."])
        samples = [
            {"sample_id": 0, "question": "Find all triangles in a graph."}
        ]
        warnings = check_anti_cheat(tree, samples, similarity_threshold=0.3)
        self.assertEqual(warnings, [])

    def test_high_overlap_flagged(self):
        """worked_example.problem 与 test sample 词序高度重叠."""
        target = "Find the number of monotone lattice paths from origin to a given point"
        cheating = "Find the number of monotone lattice paths from origin to point with coordinates"
        tree = self._make_tree([cheating])
        samples = [{"sample_id": 0, "question": target}]

        warnings = check_anti_cheat(tree, samples, similarity_threshold=0.3)
        self.assertGreater(len(warnings), 0)
        self.assertEqual(warnings[0]["node_id"], "test_concept")
        self.assertGreaterEqual(warnings[0]["jaccard"], 0.3)

    def test_empty_inputs(self):
        tree = KnowledgeTree(nodes=[])
        samples = [{"sample_id": 0, "question": "Q"}]
        warnings = check_anti_cheat(tree, samples)
        self.assertEqual(warnings, [])


# ============================================================================
# build_prompt_with_rag
# ============================================================================

class TestBuildPrompt(unittest.TestCase):

    def _make_node(self, node_id: str) -> KnowledgeNode:
        ex = WorkedExample(
            problem="problem text", solution_steps=["step 1"],
            final_answer="42", key_insight="insight",
        )
        return KnowledgeNode(
            id=node_id, title=node_id.replace("_", " ").title(),
            definition=f"Definition of {node_id}",
            key_facts=["fact 1", "fact 2"],
            worked_examples=[ex],
        )

    def test_baseline_no_nodes(self):
        """Cond A: 无 inject."""
        prompt, inject_chars = build_prompt_with_rag("Solve x+1=2", [])
        self.assertNotIn("relevant mathematical concepts", prompt)
        self.assertIn("Solve x+1=2", prompt)
        self.assertIn("\\boxed{}", prompt)
        self.assertEqual(inject_chars, 0)

    def test_with_inject_nodes(self):
        nodes = [self._make_node("binomial"), self._make_node("permutation")]
        prompt, inject_chars = build_prompt_with_rag("Count subsets", nodes)
        self.assertIn("relevant mathematical concepts", prompt)
        self.assertIn("Definition of binomial", prompt)
        self.assertIn("Definition of permutation", prompt)
        self.assertIn("Count subsets", prompt)  # 题目仍在
        self.assertGreater(inject_chars, 0)

    def test_inject_chars_count(self):
        nodes = [self._make_node("binomial")]
        _, inject_chars = build_prompt_with_rag("Q", nodes)
        # 应等于 1 个节点 llm_inject_text 长度
        self.assertEqual(inject_chars, len(nodes[0].llm_inject_text()))


# ============================================================================
# load_existing_results
# ============================================================================

class TestLoadExisting(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_nonexistent_file_returns_empty(self):
        path = os.path.join(self.tmpdir, "nope.jsonl")
        completed = load_existing_results(path)
        self.assertEqual(completed, set())

    def test_loads_completed_combinations(self):
        path = os.path.join(self.tmpdir, "results.jsonl")
        records = [
            {"sample_id": 0, "condition": "A_null", "pred_answer": "1"},
            {"sample_id": 0, "condition": "B_hybrid", "pred_answer": "2"},
            {"sample_id": 1, "condition": "A_null", "pred_answer": "3"},
        ]
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        completed = load_existing_results(path)
        self.assertEqual(completed, {
            (0, "A_null"), (0, "B_hybrid"), (1, "A_null"),
        })

    def test_malformed_lines_skipped(self):
        """坏行不应崩溃."""
        path = os.path.join(self.tmpdir, "results.jsonl")
        with open(path, "w") as f:
            f.write('{"sample_id": 0, "condition": "A_null"}\n')
            f.write('garbage line\n')
            f.write('{"missing_keys": "oops"}\n')
            f.write('{"sample_id": 1, "condition": "B_hybrid"}\n')

        completed = load_existing_results(path)
        self.assertEqual(completed, {(0, "A_null"), (1, "B_hybrid")})


if __name__ == "__main__":
    unittest.main(verbosity=2)
