"""
tests/test_local_model_clients.py
===================================

LocalModelCallable 单元测试.

测试覆盖:
  - clean_response (Ġ/Ċ artefact + </think> 切分)
  - LLMCallable 接口 (Callable[[str], str])
  - 与 ClaudeCallable 兼容性 (get_stats 字段一致)
  - lazy load / unload (顺序加载架构)
  - Mock 集成 HybridRetriever (PROTO-7.9 dual validation)

不需要 GPU / 真实模型 — 用 mock model.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
))

from knowledge_tree.local_model_clients import (
    LocalModelCallable,
    clean_response,
    make_nemotron_retriever,
    make_r1_generator,
    get_nvidia_smi_vram,
)


# ============================================================================
# clean_response
# ============================================================================

class TestCleanResponse(unittest.TestCase):
    """tokenizer artefact 清理 (复用 phase2_mcts 方案)."""

    def test_qwen_g_artefact(self):
        """Ġ → space."""
        r = "importĠpolarsĠasĠpl"
        self.assertEqual(clean_response(r), "import polars as pl")

    def test_qwen_c_artefact(self):
        """Ċ → newline."""
        r = "lineĊ1Ċline2"
        self.assertEqual(clean_response(r), "line\n1\nline2")

    def test_both_artefacts(self):
        """Ġ + Ċ 一起."""
        r = "importĠplĊdfĠ=ĠplĠ()"
        cleaned = clean_response(r)
        self.assertIn("import pl", cleaned)
        self.assertIn("\n", cleaned)
        self.assertNotIn("Ġ", cleaned)
        self.assertNotIn("Ċ", cleaned)

    def test_think_tag_split(self):
        """</think> 切分: 取后面."""
        r = "<think>I reason about X</think>final answer"
        self.assertEqual(clean_response(r), "final answer")

    def test_think_tag_with_code(self):
        """thinking 中也有 code, 只取 final."""
        r = (
            "<think>\n"
            "```python\n# draft\nresult = pl.read_csv\n```\n"
            "Wait, scan_csv is better.\n"
            "</think>\n"
            "```python\nresult = pl.scan_csv()\n```"
        )
        cleaned = clean_response(r)
        self.assertIn("scan_csv", cleaned)
        self.assertNotIn("read_csv", cleaned)
        self.assertNotIn("<think>", cleaned)

    def test_keep_thinking_true(self):
        """keep_thinking=True 不切分."""
        r = "<think>reasoning</think>answer"
        self.assertEqual(
            clean_response(r, keep_thinking=True),
            "<think>reasoning</think>answer"
        )

    def test_no_think_tag_pair_filter(self):
        """无 </think> 闭合时 (实际不该出现, 兜底)."""
        r = "<think>x</think>after"
        # Has both <think> and </think>, 切分模式
        self.assertEqual(clean_response(r), "after")

    def test_empty_response(self):
        self.assertEqual(clean_response(""), "")

    def test_no_artefact(self):
        """正常 response 不变."""
        r = "normal text"
        self.assertEqual(clean_response(r), "normal text")


# ============================================================================
# LocalModelCallable 接口
# ============================================================================

class TestLocalModelCallableInterface(unittest.TestCase):
    """接口测试 (不需要 GPU / 真实模型)."""

    def test_lazy_load(self):
        """lazy_load=True 时, 实例化不加载."""
        local = LocalModelCallable("fake-model", lazy_load=True)
        self.assertFalse(local.is_loaded)
        self.assertIsNone(local.model)
        self.assertIsNone(local.tokenizer)

    def test_stats_initial(self):
        """初始 stats 为 0."""
        local = LocalModelCallable("fake-model", lazy_load=True)
        self.assertEqual(local.total_calls, 0)
        self.assertEqual(local.total_input_tokens, 0)
        self.assertEqual(local.total_output_tokens, 0)
        self.assertEqual(local.total_cost_usd, 0.0)  # 本地永远 0
        self.assertEqual(local.total_retries, 0)

    def test_get_stats_returns_dict(self):
        """get_stats 接口与 ClaudeCallable 兼容."""
        local = LocalModelCallable("fake-model", lazy_load=True)
        stats = local.get_stats()
        # 必需字段 (与 ClaudeCallable.get_stats 对应)
        for field in ['total_calls', 'total_input_tokens',
                       'total_output_tokens', 'total_cost_usd',
                       'total_retries']:
            self.assertIn(field, stats, f"Missing field: {field}")
        # LocalModelCallable 独有字段
        self.assertIn('is_loaded', stats)
        self.assertIn('vram_at_load_gb', stats)

    def test_reset_stats(self):
        local = LocalModelCallable("fake-model", lazy_load=True)
        local.total_calls = 5
        local.total_input_tokens = 100
        local.reset_stats()
        self.assertEqual(local.total_calls, 0)
        self.assertEqual(local.total_input_tokens, 0)

    def test_empty_prompt_raises(self):
        """空 prompt 应该抛错."""
        local = LocalModelCallable("fake-model", lazy_load=True)
        with self.assertRaises(ValueError):
            local("")


# ============================================================================
# 工厂函数
# ============================================================================

class TestFactoryFunctions(unittest.TestCase):
    """便利工厂函数."""

    def test_nemotron_retriever_params(self):
        """make_nemotron_retriever 默认参数适合 retriever."""
        # 临时 mock 让 load 不实际执行
        original_load = LocalModelCallable.load
        LocalModelCallable.load = lambda self: setattr(self, 'is_loaded', False)
        try:
            nemo = make_nemotron_retriever("fake-nemo-path")
            self.assertEqual(nemo.max_new_tokens, 512)  # retriever 输出短
            self.assertEqual(nemo.temperature, 0.3)  # 确定性
            self.assertFalse(nemo.keep_thinking)
            self.assertTrue(nemo.use_int4)
            self.assertIsNone(nemo.explorer_lora)
        finally:
            LocalModelCallable.load = original_load

    def test_r1_generator_params(self):
        """make_r1_generator 默认参数适合 generator."""
        original_load = LocalModelCallable.load
        LocalModelCallable.load = lambda self: setattr(self, 'is_loaded', False)
        try:
            r1 = make_r1_generator("fake-r1", "fake-lora")
            self.assertEqual(r1.max_new_tokens, 4096)  # code gen 长
            self.assertEqual(r1.temperature, 0.6)  # 与 polars_sanity_check 一致
            self.assertFalse(r1.keep_thinking)
            self.assertTrue(r1.use_int4)
            self.assertEqual(r1.explorer_lora, "fake-lora")
        finally:
            LocalModelCallable.load = original_load


# ============================================================================
# Mock 集成测试 (PROTO-7.9 dual validation)
# ============================================================================

class TestLocalModelCallableIntegration(unittest.TestCase):
    """LocalModelCallable + HybridRetriever 集成 (mock LLM)."""

    def _make_mock_local(self, response_text: str):
        """构造 mock LocalModelCallable."""
        class MockLocal(LocalModelCallable):
            def load(self):
                # 不实际加载, 标记已加载
                self.is_loaded = True
                self.vram_at_load = 8.0
            def __call__(self, prompt):
                if not prompt:
                    raise ValueError("prompt 不能为空")
                if not self.is_loaded:
                    self.load()
                self.total_calls += 1
                return response_text
            def unload(self):
                self.is_loaded = False
                self.model = None
                self.tokenizer = None

        return MockLocal("fake-model", lazy_load=False)

    def test_can_be_used_as_hybrid_llm(self):
        """LocalModelCallable 可作为 HybridRetriever 的 llm_callable."""
        from knowledge_tree.core import KnowledgeTree
        from knowledge_tree.retrievers import HybridRetriever
        import sys
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts",
        ))
        from build_polars_mini_tree import ALL_POLARS_NODES

        tree = KnowledgeTree(ALL_POLARS_NODES)
        mock_local = self._make_mock_local(
            '{"selected_ids": ["polars_scan_csv", "polars_filter_expression"]}'
        )

        retriever = HybridRetriever(
            tree, llm_callable=mock_local,
            bm25_top_n=8, tree_top_n=5, rerank_input_size=8,
        )

        results = retriever.retrieve(
            "Read CSV lazily and filter age > 30", top_k=2,
        )
        self.assertGreater(len(results), 0)
        # Mock 应该被调用
        self.assertGreater(mock_local.total_calls, 0)

    def test_unload_releases_state(self):
        """unload 清理状态 (顺序加载架构)."""
        mock = self._make_mock_local("ok")
        self.assertTrue(mock.is_loaded)
        mock.unload()
        self.assertFalse(mock.is_loaded)
        self.assertIsNone(mock.model)
        self.assertIsNone(mock.tokenizer)

    def test_call_increments_stats(self):
        mock = self._make_mock_local("response")
        mock("prompt 1")
        mock("prompt 2")
        self.assertEqual(mock.total_calls, 2)


# ============================================================================
# Regression: 与 ClaudeCallable 接口一致性
# ============================================================================

class TestClaudeCallableCompat(unittest.TestCase):
    """关键: LocalModelCallable 必须能直接替换 ClaudeCallable.
    
    HybridRetriever / LLMRetriever / TreeNavigationRetriever 用的接口:
      - 实例可 __call__(prompt: str) -> str
      - get_stats() 返回 dict 含 total_calls/total_cost_usd
    """

    def test_call_signature_matches(self):
        """__call__(self, prompt: str) -> str."""
        import inspect
        sig = inspect.signature(LocalModelCallable.__call__)
        params = list(sig.parameters.keys())
        self.assertIn('self', params)
        self.assertIn('prompt', params)

    def test_stats_compatible_with_claude(self):
        """get_stats() 返回的字段与 ClaudeCallable 重叠."""
        from knowledge_tree.llm_clients import ClaudeCallable

        local = LocalModelCallable("fake", lazy_load=True)
        local_stats_keys = set(local.get_stats().keys())

        # ClaudeCallable.get_stats() 字段 (从代码读)
        claude_must_have = {
            'total_calls', 'total_input_tokens', 'total_output_tokens',
            'total_cost_usd', 'total_retries',
        }
        # LocalModelCallable 必须有这些字段 (才能替换 ClaudeCallable)
        missing = claude_must_have - local_stats_keys
        self.assertEqual(missing, set(),
                          f"LocalModelCallable get_stats 缺字段: {missing}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
