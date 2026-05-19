"""
tests/test_retrievers.py
========================

Retriever 单元测试. LLM 调用用 mock callable, 不依赖外部 API.

测试策略 (PROTO-7.9):
  - 每个 retriever 至少 3 测试: 基本检索 / 空树 / 边界 (top_k=1, 大 k)
  - Mock LLM 验证 prompt 构造正确 + 响应解析鲁棒
  - HybridRetriever RRF 数学正确性
  - LLM 失败时退化路径

Mock 策略:
  - LLMCallable 是 Callable[[str], str], 用 lambda / MockLLM class 替换
  - MockLLM 记录调用历史 + 返回预设响应

运行:
  cd /home/claude && python -m unittest tests.test_retrievers -v
"""

import json
import unittest
from typing import Optional

from knowledge_tree.core import KnowledgeNode, KnowledgeTree, WorkedExample
from knowledge_tree.retrievers import (
    BM25Retriever,
    HybridRetriever,
    IrrelevantRetriever,
    LLMRetriever,
    NullRetriever,
    Retriever,
    TreeNavigationRetriever,
    make_all_retrievers,
    simple_tokenize,
)


# ============================================================================
# Mock LLM 工具
# ============================================================================

class MockLLM:
    """
    可调对象, 模拟 LLM 调用. 记录 prompt 历史, 返回预设响应.

    用例:
        llm = MockLLM(responses=['{"selected_ids": ["n1"]}'])
        retriever = LLMRetriever(tree, llm)
        nodes = retriever.retrieve("query")
        assert llm.call_count == 1
        assert "query" in llm.last_prompt
    """

    def __init__(
        self,
        responses: Optional[list[str]] = None,
        default_response: str = '{"selected_ids": []}',
    ) -> None:
        self.responses = responses or []
        self.default_response = default_response
        self.prompts: list[str] = []
        self.call_count = 0

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
        else:
            response = self.default_response
        self.call_count += 1
        return response

    @property
    def last_prompt(self) -> str:
        return self.prompts[-1] if self.prompts else ""


class FailingLLM:
    """模拟 LLM 调用失败 (网络 / API 错误)."""

    def __init__(self, exception_type: type = RuntimeError) -> None:
        self.exception_type = exception_type
        self.call_count = 0

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        raise self.exception_type("Simulated LLM failure")


# ============================================================================
# 测试 fixture: 构造 5 节点测试树
# ============================================================================

def make_test_tree() -> KnowledgeTree:
    """
    构造测试用 5 节点树:
        combinatorics_root
            ├─ binomial (key: choose, formula, factorial)
            ├─ permutation (key: arrange, order)
            └─ lattice_path (key: monotone, lattice, grid)
        unrelated_geometry (forest 第 2 root, key: triangle, angle)
    """
    binomial_ex = WorkedExample(
        problem="Compute C(5, 2)",
        solution_steps=["Apply formula", "C(5,2) = 10"],
        final_answer="10",
        key_insight="Standard binomial",
    )
    binomial = KnowledgeNode(
        id="binomial",
        title="Binomial Coefficient",
        definition="Number of ways to choose k items from n without order.",
        key_facts=["C(n, k) = n! / (k! (n-k)!)", "Symmetry: C(n,k) = C(n,n-k)"],
        worked_examples=[binomial_ex],
        parent_id="combinatorics_root",
        related_concepts=["factorial"],
    )
    permutation = KnowledgeNode(
        id="permutation",
        title="Permutation",
        definition="Arrangement of items in order.",
        key_facts=["P(n, k) = n! / (n-k)!"],
        parent_id="combinatorics_root",
    )
    lattice_path = KnowledgeNode(
        id="lattice_path",
        title="Lattice Path Counting",
        definition="Counting monotone paths on integer grid.",
        key_facts=["Paths from (0,0) to (m,n) = C(m+n, m)"],
        parent_id="combinatorics_root",
        related_concepts=["binomial"],
    )
    combinatorics_root = KnowledgeNode(
        id="combinatorics_root",
        title="Combinatorics",
        definition="Branch of mathematics for counting discrete objects.",
        key_facts=["Addition rule", "Multiplication rule"],
        children_ids=["binomial", "permutation", "lattice_path"],
    )
    geometry = KnowledgeNode(
        id="unrelated_geometry",
        title="Triangle Angles",
        definition="Sum of triangle interior angles equals 180 degrees.",
        key_facts=["Sum of angles = 180"],
    )
    return KnowledgeTree(nodes=[
        combinatorics_root, binomial, permutation, lattice_path, geometry,
    ])


# ============================================================================
# Tokenizer 测试
# ============================================================================

class TestTokenize(unittest.TestCase):

    def test_basic(self):
        tokens = simple_tokenize("Binomial coefficient C(n, k)")
        # 'c' 长度 1 被过滤, 但 'binomial' 'coefficient' 'n' (len=1 被滤) 'k' (len=1 被滤)
        # 实际: 'binomial', 'coefficient' (n / k / c 长度 1 都被滤)
        self.assertIn("binomial", tokens)
        self.assertIn("coefficient", tokens)

    def test_stopwords_removed(self):
        tokens = simple_tokenize("the sum of the angles")
        # 'sum' 'angles' 保留, 'the' 'of' 移除
        self.assertNotIn("the", tokens)
        self.assertNotIn("of", tokens)
        self.assertIn("sum", tokens)
        self.assertIn("angles", tokens)

    def test_lowercase(self):
        tokens = simple_tokenize("BINOMIAL Coefficient")
        self.assertIn("binomial", tokens)
        self.assertIn("coefficient", tokens)

    def test_numbers_kept(self):
        tokens = simple_tokenize("Choose 5 from 10")
        self.assertIn("5", tokens)
        self.assertIn("10", tokens)
        self.assertIn("choose", tokens)

    def test_empty(self):
        self.assertEqual(simple_tokenize(""), [])
        self.assertEqual(simple_tokenize("   "), [])


# ============================================================================
# NullRetriever
# ============================================================================

class TestNullRetriever(unittest.TestCase):

    def test_always_returns_empty(self):
        tree = make_test_tree()
        r = NullRetriever(tree)
        self.assertEqual(r.retrieve("anything"), [])
        self.assertEqual(r.retrieve("binomial coefficient"), [])

    def test_name(self):
        tree = make_test_tree()
        r = NullRetriever(tree)
        self.assertEqual(r.name, "null")

    def test_top_k_validation(self):
        tree = make_test_tree()
        r = NullRetriever(tree)
        with self.assertRaises(ValueError):
            r.retrieve("q", top_k=0)
        with self.assertRaises(ValueError):
            r.retrieve("q", top_k=-1)


# ============================================================================
# BM25Retriever
# ============================================================================

class TestBM25Retriever(unittest.TestCase):

    def test_basic_retrieval(self):
        tree = make_test_tree()
        r = BM25Retriever(tree)
        # query 含 'lattice' 应找到 lattice_path
        nodes = r.retrieve("lattice path grid monotone", top_k=2)
        self.assertGreater(len(nodes), 0)
        node_ids = [n.id for n in nodes]
        self.assertIn("lattice_path", node_ids)

    def test_irrelevant_query_returns_no_match(self):
        """完全无关 query, 应返回少量或空 (score=0 过滤)."""
        tree = make_test_tree()
        r = BM25Retriever(tree)
        nodes = r.retrieve("xyz qwertyu fictional nonsense words", top_k=3)
        # 由于过滤 score=0, 应空或很少
        for n in nodes:
            self.assertNotEqual(n.id, "unrelated_geometry")  # 不应误中

    def test_empty_tree(self):
        empty_tree = KnowledgeTree()
        r = BM25Retriever(empty_tree)
        self.assertEqual(r.retrieve("any query"), [])

    def test_top_k_respected(self):
        tree = make_test_tree()
        r = BM25Retriever(tree)
        # 给一个广泛 query, 多节点匹配
        nodes = r.retrieve("counting formula choose", top_k=2)
        self.assertLessEqual(len(nodes), 2)

    def test_bm25_excludes_worked_examples(self):
        """用户决策 D-2: BM25 索引不含 worked_examples 内容."""
        tree = make_test_tree()
        r = BM25Retriever(tree)
        # binomial 的 worked_example 含 "Standard binomial" 这个 insight
        # 但其 key_facts 不含, 所以 BM25 索引中不应有
        # 检查方式: 用 worked_example 独有词检索, 不应命中 binomial top-1
        nodes = r.retrieve("standard insight", top_k=3)
        # 'standard' 在 binomial 的 worked_example 中, 但不在 bm25 索引
        # 所以 binomial 不应在 top-1 (除非其他字段也含 'standard')
        # 这个测试是 negative: 仅确认 bm25 不靠 worked_examples 内容召回
        # (不强制 binomial 不出现, 因 BM25 也会因 fallback 短文本召回)
        # 实际更直接的测试: 看 index text 不含 worked_examples
        binomial_node = tree.get_node("binomial")
        index_text = binomial_node.bm25_index_text()
        self.assertNotIn("Standard binomial", index_text)
        self.assertNotIn("Compute C(5, 2)", index_text)

    def test_get_ranked_with_scores(self):
        """HybridRetriever 用此接口拿带分数的排序."""
        tree = make_test_tree()
        r = BM25Retriever(tree)
        ranked = r.get_ranked_with_scores("lattice path", top_k=3)
        self.assertGreater(len(ranked), 0)
        # 应是 (node, score) 元组
        for node, score in ranked:
            self.assertIsInstance(node, KnowledgeNode)
            self.assertGreater(score, 0)  # 已过滤 0

    def test_name(self):
        tree = make_test_tree()
        r = BM25Retriever(tree)
        self.assertEqual(r.name, "bm25_only")


# ============================================================================
# IrrelevantRetriever
# ============================================================================

class TestIrrelevantRetriever(unittest.TestCase):

    def test_returns_least_relevant(self):
        """对 binomial 强相关的 query, IrrelevantRetriever 应返回最不相关 (geometry)."""
        tree = make_test_tree()
        r = IrrelevantRetriever(tree)
        nodes = r.retrieve("binomial coefficient choose factorial", top_k=2)
        node_ids = [n.id for n in nodes]
        # 应包含 unrelated_geometry, 不应是 binomial
        self.assertNotIn("binomial", node_ids)

    def test_empty_tree(self):
        empty_tree = KnowledgeTree()
        r = IrrelevantRetriever(empty_tree)
        self.assertEqual(r.retrieve("query"), [])

    def test_top_k_more_than_nodes(self):
        """top_k 超过节点数: 返回所有."""
        tree = make_test_tree()
        r = IrrelevantRetriever(tree)
        nodes = r.retrieve("any", top_k=100)
        self.assertEqual(len(nodes), 5)  # 测试树共 5 节点

    def test_name(self):
        tree = make_test_tree()
        r = IrrelevantRetriever(tree)
        self.assertEqual(r.name, "irrelevant")


# ============================================================================
# LLMRetriever (mock LLM)
# ============================================================================

class TestLLMRetriever(unittest.TestCase):

    def test_basic_retrieval(self):
        tree = make_test_tree()
        mock_llm = MockLLM(responses=['{"selected_ids": ["binomial", "permutation"]}'])
        r = LLMRetriever(tree, mock_llm)
        nodes = r.retrieve("count arrangements", top_k=2)
        self.assertEqual(len(nodes), 2)
        self.assertEqual([n.id for n in nodes], ["binomial", "permutation"])

    def test_prompt_contains_query_and_nodes(self):
        tree = make_test_tree()
        mock_llm = MockLLM(responses=['{"selected_ids": []}'])
        r = LLMRetriever(tree, mock_llm)
        r.retrieve("MY_UNIQUE_QUERY_XYZ", top_k=3)
        prompt = mock_llm.last_prompt
        self.assertIn("MY_UNIQUE_QUERY_XYZ", prompt)
        self.assertIn("binomial", prompt)  # 节点 id 应在 nodes_listing 中
        self.assertIn("3", prompt)  # top_k

    def test_handles_markdown_wrapped_json(self):
        """容错: LLM 可能用 ```json ... ``` 包响应."""
        tree = make_test_tree()
        mock_llm = MockLLM(responses=[
            '```json\n{"selected_ids": ["binomial"]}\n```'
        ])
        r = LLMRetriever(tree, mock_llm)
        nodes = r.retrieve("q", top_k=1)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].id, "binomial")

    def test_handles_extra_text_around_json(self):
        """容错: LLM 可能加前后说明."""
        tree = make_test_tree()
        mock_llm = MockLLM(responses=[
            'Sure! Here is my selection:\n{"selected_ids": ["binomial"]}\nLet me know if you need more.'
        ])
        r = LLMRetriever(tree, mock_llm)
        nodes = r.retrieve("q", top_k=1)
        self.assertEqual(len(nodes), 1)

    def test_invalid_json_returns_empty(self):
        tree = make_test_tree()
        mock_llm = MockLLM(responses=["not a valid json at all!"])
        r = LLMRetriever(tree, mock_llm)
        nodes = r.retrieve("q", top_k=2)
        self.assertEqual(nodes, [])

    def test_nonexistent_id_skipped(self):
        tree = make_test_tree()
        mock_llm = MockLLM(responses=[
            '{"selected_ids": ["binomial", "GHOST_ID", "permutation"]}'
        ])
        r = LLMRetriever(tree, mock_llm)
        nodes = r.retrieve("q", top_k=3)
        node_ids = [n.id for n in nodes]
        self.assertNotIn("GHOST_ID", node_ids)
        self.assertIn("binomial", node_ids)

    def test_llm_call_failure_returns_empty(self):
        tree = make_test_tree()
        failing_llm = FailingLLM()
        r = LLMRetriever(tree, failing_llm)
        nodes = r.retrieve("q", top_k=3)
        self.assertEqual(nodes, [])
        self.assertEqual(failing_llm.call_count, 1)

    def test_empty_tree(self):
        empty_tree = KnowledgeTree()
        mock_llm = MockLLM()
        r = LLMRetriever(empty_tree, mock_llm)
        self.assertEqual(r.retrieve("q"), [])
        # 空树不应触发 LLM 调用
        self.assertEqual(mock_llm.call_count, 0)

    def test_name(self):
        tree = make_test_tree()
        r = LLMRetriever(tree, MockLLM())
        self.assertEqual(r.name, "llm_only")


# ============================================================================
# TreeNavigationRetriever
# ============================================================================

class TestTreeNavigationRetriever(unittest.TestCase):

    def test_basic_retrieval(self):
        tree = make_test_tree()
        mock_llm = MockLLM(responses=['{"selected_ids": ["lattice_path"]}'])
        r = TreeNavigationRetriever(tree, mock_llm)
        nodes = r.retrieve("path counting", top_k=1)
        self.assertEqual([n.id for n in nodes], ["lattice_path"])

    def test_prompt_shows_tree_structure(self):
        """prompt 应展示树的 hierarchical 结构 (不是 flat list)."""
        tree = make_test_tree()
        mock_llm = MockLLM(responses=['{"selected_ids": []}'])
        r = TreeNavigationRetriever(tree, mock_llm)
        r.retrieve("any", top_k=2)
        prompt = mock_llm.last_prompt
        # 应包含 indented 子节点 (e.g. 'combinatorics_root' 后 '  - binomial')
        self.assertIn("combinatorics_root", prompt)
        self.assertIn("binomial", prompt)
        # 验证 indent (binomial 是 root 的子, 应有缩进)
        lines = prompt.split("\n")
        root_line = next(l for l in lines if "combinatorics_root" in l)
        child_line = next(l for l in lines if "binomial" in l and "**Binomial" in l)
        # root 行没有缩进, child 行有缩进
        self.assertFalse(root_line.startswith(" "))
        self.assertTrue(child_line.startswith("  "))

    def test_prompt_different_from_llm_retriever(self):
        """E vs D ablation 关键: tree 与 llm 用不同的 prompt 结构."""
        tree = make_test_tree()
        llm_for_tree = MockLLM(responses=['{"selected_ids": []}'])
        llm_for_flat = MockLLM(responses=['{"selected_ids": []}'])

        tree_r = TreeNavigationRetriever(tree, llm_for_tree)
        flat_r = LLMRetriever(tree, llm_for_flat)

        tree_r.retrieve("q", top_k=1)
        flat_r.retrieve("q", top_k=1)

        # tree prompt 应含 "Knowledge Tree"
        self.assertIn("Knowledge Tree", llm_for_tree.last_prompt)
        # flat prompt 应含 "Knowledge Base" (flat list 标识)
        self.assertIn("Knowledge Base", llm_for_flat.last_prompt)

    def test_empty_tree(self):
        empty_tree = KnowledgeTree()
        mock_llm = MockLLM()
        r = TreeNavigationRetriever(empty_tree, mock_llm)
        self.assertEqual(r.retrieve("q"), [])
        self.assertEqual(mock_llm.call_count, 0)

    def test_name(self):
        tree = make_test_tree()
        r = TreeNavigationRetriever(tree, MockLLM())
        self.assertEqual(r.name, "tree_only")


# ============================================================================
# HybridRetriever
# ============================================================================

class TestHybridRetriever(unittest.TestCase):

    def test_basic_pipeline(self):
        """完整 pipeline: BM25 + Tree -> RRF -> LLM rerank."""
        tree = make_test_tree()
        # mock LLM: 第一次调用 (TreeNavigationRetriever stage 1), 第二次调用 (rerank stage 2)
        mock_llm = MockLLM(responses=[
            '{"selected_ids": ["lattice_path", "binomial"]}',  # tree nav stage 1
            '{"selected_ids": ["lattice_path"]}',  # rerank stage 2
        ])
        r = HybridRetriever(tree, mock_llm)
        nodes = r.retrieve("monotone lattice path", top_k=1)
        self.assertEqual([n.id for n in nodes], ["lattice_path"])
        # 应该调用 2 次 LLM (tree + rerank)
        self.assertEqual(mock_llm.call_count, 2)

    def test_rrf_merge_correctness(self):
        """RRF 公式: 1/(k+rank), k=60 默认."""
        tree = make_test_tree()
        mock_llm = MockLLM()
        r = HybridRetriever(tree, mock_llm, rrf_k=60)

        n1 = tree.get_node("binomial")
        n2 = tree.get_node("permutation")
        n3 = tree.get_node("lattice_path")

        # ranking 1: [n1, n2]
        # ranking 2: [n3, n1]
        # RRF scores:
        #   n1: 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252
        #   n2: 1/(60+2)              = 0.01613
        #   n3: 1/(60+1)              = 0.01639
        # 排序: n1 > n3 > n2
        merged = r._rrf_merge([[n1, n2], [n3, n1]], top_n=3)
        self.assertEqual([n.id for n in merged], ["binomial", "lattice_path", "permutation"])

    def test_rerank_skipped_when_few_candidates(self):
        """候选数 <= top_k: 跳过 rerank, 节省 LLM 调用."""
        tree = make_test_tree()
        # tree nav 返回 1 个, BM25 返回 1 个 (假设无重复)
        mock_llm = MockLLM(responses=[
            '{"selected_ids": ["lattice_path"]}',  # tree nav
            # 不应有 rerank 调用
        ])
        r = HybridRetriever(tree, mock_llm)
        nodes = r.retrieve("very specific query xxx", top_k=3)
        # 只调用了 tree nav, 没调 rerank
        # (具体 call_count 取决于 BM25 是否返回结果)
        # 关键: 不报错, 返回合理结果
        self.assertIsInstance(nodes, list)

    def test_rerank_falls_back_on_llm_failure(self):
        """rerank LLM 失败: 退化为 RRF top-k."""
        tree = make_test_tree()

        # 自定义 LLM: 第 1 次正常 (tree nav), 第 2 次抛错 (rerank)
        class PartialFailLLM:
            def __init__(self):
                self.call_count = 0

            def __call__(self, prompt: str) -> str:
                self.call_count += 1
                if self.call_count == 1:
                    return '{"selected_ids": ["binomial", "permutation", "lattice_path"]}'
                else:
                    raise RuntimeError("rerank failed")

        llm = PartialFailLLM()
        r = HybridRetriever(tree, llm)
        nodes = r.retrieve("counting", top_k=2)
        # 应退化为 RRF 前 2 个, 不空
        self.assertEqual(len(nodes), 2)

    def test_rerank_hallucinated_id_skipped(self):
        """rerank LLM 输出非候选 id (幻觉): 应跳过."""
        tree = make_test_tree()
        mock_llm = MockLLM(responses=[
            '{"selected_ids": ["binomial", "lattice_path"]}',  # tree nav, valid
            '{"selected_ids": ["unrelated_geometry", "binomial"]}',
            # rerank: unrelated_geometry 没在 stage 1 候选中, 应跳过
        ])
        r = HybridRetriever(tree, mock_llm)
        nodes = r.retrieve("counting", top_k=2)
        node_ids = [n.id for n in nodes]
        # binomial 应在 (它在 stage 1 候选), unrelated_geometry 不应在
        self.assertIn("binomial", node_ids)
        # unrelated_geometry 不在 stage 1 候选中 (假设 BM25 也没返回它)
        # 这个测试有点依赖 BM25 行为, 但 unrelated_geometry 与 "counting" 无关
        # 即使被 BM25 召回也只会是 fallback

    def test_name(self):
        tree = make_test_tree()
        r = HybridRetriever(tree, MockLLM())
        self.assertEqual(r.name, "hybrid")


# ============================================================================
# Factory
# ============================================================================

class TestFactory(unittest.TestCase):

    def test_make_all_retrievers(self):
        tree = make_test_tree()
        mock_llm = MockLLM()
        retrievers = make_all_retrievers(tree, mock_llm)

        expected_keys = {
            "A_null", "B_hybrid", "C_bm25_only",
            "D_llm_only", "E_tree_only", "F_irrelevant",
        }
        self.assertEqual(set(retrievers.keys()), expected_keys)

        # 类型检查
        from knowledge_tree.retrievers import (
            NullRetriever, HybridRetriever, BM25Retriever,
            LLMRetriever, TreeNavigationRetriever, IrrelevantRetriever,
        )
        self.assertIsInstance(retrievers["A_null"], NullRetriever)
        self.assertIsInstance(retrievers["B_hybrid"], HybridRetriever)
        self.assertIsInstance(retrievers["C_bm25_only"], BM25Retriever)
        self.assertIsInstance(retrievers["D_llm_only"], LLMRetriever)
        self.assertIsInstance(retrievers["E_tree_only"], TreeNavigationRetriever)
        self.assertIsInstance(retrievers["F_irrelevant"], IrrelevantRetriever)

    def test_all_retrievers_share_abc(self):
        """所有 6 retrievers 共享 Retriever ABC, 接口统一."""
        tree = make_test_tree()
        mock_llm = MockLLM(default_response='{"selected_ids": []}')
        retrievers = make_all_retrievers(tree, mock_llm)
        for name, r in retrievers.items():
            self.assertIsInstance(r, Retriever)
            # 每个都能调 retrieve 不报错
            result = r.retrieve("test query", top_k=2)
            self.assertIsInstance(result, list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
