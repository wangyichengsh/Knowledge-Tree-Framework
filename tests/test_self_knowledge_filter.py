"""
tests/test_self_knowledge_filter.py
=====================================

Stage 1.3c: SelfKnowledgeFilter + HybridWithSelfKnowledgeRetriever 单元测试.

不需要 GPU / 真实模型 (用 mock LLMCallable).

测试覆盖:
  - SelfKnowledgeFilter._parse_decision (robust 解析)
  - SelfKnowledgeFilter.filter (核心逻辑 + min_keep fallback)
  - HybridWithSelfKnowledgeRetriever (集成)
  - Stats correctness
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
))

from knowledge_tree.self_knowledge_filter import (
    SelfKnowledgeFilter,
    HybridWithSelfKnowledgeRetriever,
    DEFAULT_SELF_KNOWLEDGE_PROMPT,
)


# ============================================================================
# _parse_decision
# ============================================================================

class TestParseDecision(unittest.TestCase):
    """Robust 解析 LLM 响应为 YES/NO."""

    def test_yes_clean(self):
        self.assertEqual(SelfKnowledgeFilter._parse_decision("YES"), "YES")

    def test_no_clean(self):
        self.assertEqual(SelfKnowledgeFilter._parse_decision("NO"), "NO")

    def test_yes_lowercase(self):
        self.assertEqual(SelfKnowledgeFilter._parse_decision("yes"), "YES")

    def test_yes_with_explanation(self):
        """模型有时多说一句."""
        r = "YES, I already know how to use scan_csv."
        self.assertEqual(SelfKnowledgeFilter._parse_decision(r), "YES")

    def test_no_with_explanation(self):
        r = "NO, I need this reference for the Polars 1.0+ API."
        self.assertEqual(SelfKnowledgeFilter._parse_decision(r), "NO")

    def test_empty_defaults_no(self):
        """空响应 fallback NO (保留节点)."""
        self.assertEqual(SelfKnowledgeFilter._parse_decision(""), "NO")

    def test_garbage_defaults_no(self):
        """无 YES/NO token fallback NO."""
        self.assertEqual(SelfKnowledgeFilter._parse_decision("hmm idk"), "NO")

    def test_yes_in_other_words_does_not_match(self):
        """注意: regex \\bYES\\b 必须是独立词."""
        self.assertEqual(SelfKnowledgeFilter._parse_decision("YESTERDAY"), "NO")

    def test_no_first_priority(self):
        """如果 NO 在前 YES 在后, 取 NO (regex.search 返回第一个 match)."""
        r = "NO, I'm not sure if YES applies here"
        self.assertEqual(SelfKnowledgeFilter._parse_decision(r), "NO")


# ============================================================================
# SelfKnowledgeFilter
# ============================================================================

class TestSelfKnowledgeFilter(unittest.TestCase):

    def _make_node(self, node_id="n1", title="N1"):
        from knowledge_tree.core import KnowledgeNode, WorkedExample
        return KnowledgeNode(
            id=node_id,
            title=title,
            definition=f"Definition of {node_id}",
            key_facts=["fact 1"],
            worked_examples=[WorkedExample(
                problem="p", solution_steps=["s"], final_answer="a",
            )],
            common_pitfalls=["DO NOT X"],
        )

    def test_filter_all_no_keeps_all(self):
        """LLM 全说 NO → 全保留."""
        mock_llm = MagicMock(return_value="NO")
        filter = SelfKnowledgeFilter(mock_llm)
        
        nodes = [self._make_node(f"n{i}") for i in range(3)]
        kept = filter.filter("query", nodes)
        
        self.assertEqual(len(kept), 3)
        self.assertEqual(filter.total_kept, 3)
        self.assertEqual(filter.total_filtered_out, 0)

    def test_filter_all_yes_keeps_min(self):
        """LLM 全说 YES → fallback 保留 min_keep."""
        mock_llm = MagicMock(return_value="YES")
        filter = SelfKnowledgeFilter(mock_llm, min_keep=1)
        
        nodes = [self._make_node(f"n{i}") for i in range(3)]
        kept = filter.filter("query", nodes)
        
        # min_keep=1 fallback
        self.assertEqual(len(kept), 1)
        # 应该是第一个 (按 retrieval order)
        self.assertEqual(kept[0].id, "n0")

    def test_filter_mixed(self):
        """部分 YES 部分 NO."""
        responses = ["YES", "NO", "YES", "NO"]
        mock_llm = MagicMock(side_effect=responses)
        filter = SelfKnowledgeFilter(mock_llm)
        
        nodes = [self._make_node(f"n{i}") for i in range(4)]
        kept = filter.filter("query", nodes)
        
        # n1, n3 NO 应保留
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0].id, "n1")
        self.assertEqual(kept[1].id, "n3")
        self.assertEqual(filter.total_filtered_out, 2)

    def test_filter_empty_input(self):
        """空 input → 空 output."""
        mock_llm = MagicMock()
        filter = SelfKnowledgeFilter(mock_llm)
        
        kept = filter.filter("query", [])
        
        self.assertEqual(kept, [])
        mock_llm.assert_not_called()

    def test_llm_error_fallback_keep(self):
        """LLM 调用失败 → fallback keep node."""
        mock_llm = MagicMock(side_effect=RuntimeError("API down"))
        filter = SelfKnowledgeFilter(mock_llm)
        
        nodes = [self._make_node(f"n{i}") for i in range(2)]
        kept = filter.filter("query", nodes)
        
        # 全 fallback 保留
        self.assertEqual(len(kept), 2)
        self.assertEqual(filter.total_errors, 2)

    def test_min_keep_zero(self):
        """min_keep=0 允许全过滤."""
        mock_llm = MagicMock(return_value="YES")
        filter = SelfKnowledgeFilter(mock_llm, min_keep=0)
        
        nodes = [self._make_node(f"n{i}") for i in range(3)]
        kept = filter.filter("query", nodes)
        
        self.assertEqual(len(kept), 0)

    def test_stats_correctness(self):
        responses = ["YES", "NO", "YES"]
        mock_llm = MagicMock(side_effect=responses)
        filter = SelfKnowledgeFilter(mock_llm, min_keep=0)
        
        nodes = [self._make_node(f"n{i}") for i in range(3)]
        filter.filter("query", nodes)
        
        stats = filter.get_stats()
        self.assertEqual(stats['total_queried'], 3)
        self.assertEqual(stats['total_kept'], 1)
        self.assertEqual(stats['total_filtered_out'], 2)
        self.assertEqual(stats['filter_rate'], 2/3)

    def test_reset_stats(self):
        mock_llm = MagicMock(return_value="NO")
        filter = SelfKnowledgeFilter(mock_llm)
        filter.filter("q", [self._make_node()])
        filter.reset_stats()
        
        self.assertEqual(filter.total_queried, 0)


# ============================================================================
# HybridWithSelfKnowledgeRetriever (integration)
# ============================================================================

class TestHybridWithSelfKnowledgeRetriever(unittest.TestCase):

    def _load_tree(self):
        """加载真实 Polars KTF (32 节点)."""
        try:
            from build_polars_mini_tree import ALL_POLARS_NODES
            from knowledge_tree.core import KnowledgeTree
            return KnowledgeTree(ALL_POLARS_NODES)
        except ImportError:
            self.skipTest("build_polars_mini_tree not available")

    def test_full_pipeline(self):
        """端到端 (mock 两个 LLM)."""
        tree = self._load_tree()
        
        # Mock rerank LLM (HybridRetriever 用)
        rerank_llm = MagicMock(return_value=(
            '{"selected_ids": ["polars_scan_csv", "polars_filter_expression", '
            '"polars_lazyframe", "polars_collect", "polars_with_columns"]}'
        ))
        # Mock filter LLM (主模型, 全说 NO 表示需要)
        filter_llm = MagicMock(return_value="NO")
        
        retriever = HybridWithSelfKnowledgeRetriever(
            tree, rerank_llm=rerank_llm, filter_llm=filter_llm,
            top_k_initial=5,
        )
        
        results = retriever.retrieve("Read CSV lazily", top_k=3)
        
        # 应有 3 个节点 (全保留, 但 top_k=3 限制)
        self.assertEqual(len(results), 3)
        # filter 被调 5 次 (initial top-5)
        self.assertEqual(filter_llm.call_count, 5)

    def test_filter_reduces_inject(self):
        """Filter 成功减少 inject 数量."""
        tree = self._load_tree()
        
        rerank_llm = MagicMock(return_value=(
            '{"selected_ids": ["polars_scan_csv", "polars_filter_expression", '
            '"polars_lazyframe", "polars_collect", "polars_with_columns"]}'
        ))
        # filter 说 YES YES NO NO YES → 仅 2 个 NO 保留
        filter_responses = ["YES", "YES", "NO", "NO", "YES"]
        filter_llm = MagicMock(side_effect=filter_responses)
        
        retriever = HybridWithSelfKnowledgeRetriever(
            tree, rerank_llm=rerank_llm, filter_llm=filter_llm,
            top_k_initial=5,
        )
        
        results = retriever.retrieve("Read CSV lazily", top_k=3)
        
        # 仅 2 NO 保留 (top_k=3 但实际 <3)
        self.assertEqual(len(results), 2)

    def test_min_keep_prevents_empty(self):
        """All YES + min_keep=1 → 保留 top-1."""
        tree = self._load_tree()
        
        rerank_llm = MagicMock(return_value=(
            '{"selected_ids": ["polars_scan_csv", "polars_filter_expression"]}'
        ))
        filter_llm = MagicMock(return_value="YES")
        
        retriever = HybridWithSelfKnowledgeRetriever(
            tree, rerank_llm=rerank_llm, filter_llm=filter_llm,
            top_k_initial=2, min_keep=1,
        )
        
        results = retriever.retrieve("Read CSV", top_k=3)
        
        # min_keep=1, 保留 1 个 (即使全说 YES)
        self.assertEqual(len(results), 1)

    def test_retriever_name(self):
        tree = self._load_tree()
        rerank_llm = MagicMock(return_value='{"selected_ids": []}')
        filter_llm = MagicMock(return_value="NO")
        retriever = HybridWithSelfKnowledgeRetriever(
            tree, rerank_llm=rerank_llm, filter_llm=filter_llm,
        )
        self.assertEqual(retriever.name, "hybrid_with_self_knowledge")

    def test_stats_propagation(self):
        """get_stats 返回 filter 统计."""
        tree = self._load_tree()
        
        rerank_llm = MagicMock(return_value=(
            '{"selected_ids": ["polars_scan_csv", "polars_filter_expression"]}'
        ))
        filter_llm = MagicMock(side_effect=["YES", "NO"])
        
        retriever = HybridWithSelfKnowledgeRetriever(
            tree, rerank_llm=rerank_llm, filter_llm=filter_llm,
            top_k_initial=2, min_keep=0,
        )
        retriever.retrieve("query", top_k=2)
        
        stats = retriever.get_stats()
        self.assertEqual(stats['total_queried'], 2)
        self.assertEqual(stats['retriever_name'], "hybrid_with_self_knowledge")


if __name__ == "__main__":
    unittest.main(verbosity=2)
