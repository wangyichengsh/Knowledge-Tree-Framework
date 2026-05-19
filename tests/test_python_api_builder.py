"""
tests/test_python_api_builder.py
==================================

Phase 4.2 Stage 1.3b: PythonAPIBuilder 单元测试.

测试覆盖:
  - parse_api_name (含 namespace 处理)
  - resolve_api_object (Module / class / namespace)
  - parse_numpy_docstring (sections)
  - extract_doctest_examples
  - safe_signature (含 pl.col 特殊情况)
  - PythonAPIBuilder full build
  - 不需要 LLM (introspection-only path)
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Parsing helpers
# ============================================================================

class TestParseAPIName(unittest.TestCase):
    """parse_api_name: API name -> (parts, qualified)."""

    def test_module_level(self):
        from knowledge_tree.python_api_builder import parse_api_name
        parts, qual = parse_api_name("pl.scan_csv")
        self.assertEqual(qual, "polars.scan_csv")

    def test_class_method(self):
        from knowledge_tree.python_api_builder import parse_api_name
        parts, qual = parse_api_name("DataFrame.pivot")
        self.assertEqual(qual, "polars.DataFrame.pivot")

    def test_namespace_method(self):
        from knowledge_tree.python_api_builder import parse_api_name
        parts, qual = parse_api_name("Expr.str.contains")
        self.assertEqual(qual, "polars.Expr.str.contains")

    def test_full_polars_prefix(self):
        from knowledge_tree.python_api_builder import parse_api_name
        parts, qual = parse_api_name("polars.scan_csv")
        self.assertEqual(qual, "polars.scan_csv")


class TestResolveAPIObject(unittest.TestCase):
    """resolve_api_object: parts -> Python callable."""

    def test_module_function(self):
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars not installed")
        from knowledge_tree.python_api_builder import (
            parse_api_name, resolve_api_object,
        )
        parts, _ = parse_api_name("pl.scan_csv")
        obj = resolve_api_object(parts, pl)
        self.assertIs(obj, pl.scan_csv)

    def test_class_method(self):
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars not installed")
        from knowledge_tree.python_api_builder import (
            parse_api_name, resolve_api_object,
        )
        parts, _ = parse_api_name("DataFrame.pivot")
        obj = resolve_api_object(parts, pl)
        self.assertTrue(callable(obj))

    def test_namespace_method(self):
        """关键: namespace 通过 property 访问"""
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars not installed")
        from knowledge_tree.python_api_builder import (
            parse_api_name, resolve_api_object,
        )
        parts, _ = parse_api_name("Expr.str.contains")
        obj = resolve_api_object(parts, pl)
        self.assertTrue(callable(obj))


# ============================================================================
# Docstring parsing
# ============================================================================

class TestParseNumpyDocstring(unittest.TestCase):
    """parse_numpy_docstring: numpy/pandas 风格 sections."""

    def test_description_only(self):
        from knowledge_tree.python_api_builder import parse_numpy_docstring
        sections = parse_numpy_docstring("Lazily read CSV.")
        self.assertEqual(sections["Description"], "Lazily read CSV.")

    def test_with_parameters(self):
        from knowledge_tree.python_api_builder import parse_numpy_docstring
        doc = """Read CSV.

Parameters
----------
source : str
    File path.

Returns
-------
LazyFrame
"""
        sections = parse_numpy_docstring(doc)
        self.assertIn("Parameters", sections)
        self.assertIn("Returns", sections)
        self.assertIn("source", sections["Parameters"])

    def test_empty(self):
        from knowledge_tree.python_api_builder import parse_numpy_docstring
        sections = parse_numpy_docstring("")
        self.assertEqual(sections, {"Description": ""})


class TestExtractDoctest(unittest.TestCase):
    """提取 >>> 示例."""

    def test_single_example(self):
        from knowledge_tree.python_api_builder import extract_doctest_examples
        doc = """Desc.

>>> df = pl.DataFrame({"a": [1]})
shape: (1, 1)
"""
        examples = extract_doctest_examples(doc)
        self.assertEqual(len(examples), 1)
        self.assertIn("DataFrame", examples[0]["code"])

    def test_no_example(self):
        from knowledge_tree.python_api_builder import extract_doctest_examples
        examples = extract_doctest_examples("Just description.")
        self.assertEqual(examples, [])

    def test_multi_examples(self):
        from knowledge_tree.python_api_builder import extract_doctest_examples
        doc = """Desc.

>>> a = 1
1

>>> b = 2
2
"""
        examples = extract_doctest_examples(doc)
        self.assertGreaterEqual(len(examples), 2)


class TestSafeSignature(unittest.TestCase):
    """safe_signature: robust 处理特殊情况."""

    def test_normal_function(self):
        from knowledge_tree.python_api_builder import safe_signature
        def f(x: int, y: str = "a") -> bool:
            return True
        sig = safe_signature(f)
        self.assertIn("x", sig)
        self.assertIn("y", sig)

    def test_no_signature_fallback(self):
        from knowledge_tree.python_api_builder import safe_signature
        # 自定义类不实现 __signature__
        class Weird:
            def __getattr__(self, name):
                raise TypeError("no")
        # safe_signature should return "(...)" not raise
        sig = safe_signature(Weird())
        self.assertEqual(sig, "(...)")


# ============================================================================
# Full build
# ============================================================================

class TestPythonAPIBuilder(unittest.TestCase):
    """PythonAPIBuilder full build."""

    def test_build_single_api(self):
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars not installed")
        from knowledge_tree.python_api_builder import PythonAPIBuilder

        builder = PythonAPIBuilder()
        nodes = builder.build_from_api_list(["pl.scan_csv"], skip_on_failure=False)
        self.assertEqual(len(nodes), 1)
        node = nodes[0]
        self.assertEqual(node.id, "polars_scan_csv")
        self.assertIn("scan_csv", node.title)
        self.assertTrue(node.definition)
        self.assertGreater(len(node.key_facts), 0)
        self.assertGreater(len(node.worked_examples), 0)
        # Metadata
        self.assertEqual(node.domain_metadata['api_name'], "pl.scan_csv")
        self.assertEqual(node.domain_metadata['qualified_name'], "polars.scan_csv")

    def test_build_namespace_method(self):
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars not installed")
        from knowledge_tree.python_api_builder import PythonAPIBuilder

        builder = PythonAPIBuilder()
        nodes = builder.build_from_api_list(
            ["Expr.str.contains"], skip_on_failure=False,
        )
        self.assertEqual(len(nodes), 1)
        self.assertIn("contains", nodes[0].id)

    def test_pitfall_auto_detect_cum_sum(self):
        """cum_sum 应自动检测出 Polars 1.0+ 重命名 pitfall."""
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars not installed")
        from knowledge_tree.python_api_builder import PythonAPIBuilder

        builder = PythonAPIBuilder()
        nodes = builder.build_from_api_list(["Expr.cum_sum"], skip_on_failure=False)
        self.assertEqual(len(nodes), 1)
        # 应有 0.x 重命名 pitfall
        self.assertTrue(
            any("0.x" in p or "underscore" in p.lower() for p in nodes[0].common_pitfalls),
            f"Expected pitfall about 0.x naming, got: {nodes[0].common_pitfalls}"
        )

    def test_pitfall_auto_detect_unpivot(self):
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars not installed")
        from knowledge_tree.python_api_builder import PythonAPIBuilder

        builder = PythonAPIBuilder()
        nodes = builder.build_from_api_list(["DataFrame.unpivot"], skip_on_failure=False)
        self.assertEqual(len(nodes), 1)
        # 应有 .melt deprecated pitfall
        self.assertTrue(
            any("melt" in p.lower() for p in nodes[0].common_pitfalls),
            f"Expected pitfall about .melt, got: {nodes[0].common_pitfalls}"
        )

    def test_skip_on_failure(self):
        """不存在的 API 应该 skip."""
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars not installed")
        from knowledge_tree.python_api_builder import PythonAPIBuilder

        builder = PythonAPIBuilder()
        nodes = builder.build_from_api_list(
            ["pl.scan_csv", "pl.nonexistent_xyz"],
            skip_on_failure=True,
        )
        self.assertEqual(len(nodes), 1)  # 仅 scan_csv 成功
        self.assertEqual(builder.total_built, 1)
        self.assertEqual(builder.total_failed, 1)

    def test_stats(self):
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars not installed")
        from knowledge_tree.python_api_builder import PythonAPIBuilder

        builder = PythonAPIBuilder()
        builder.build_from_api_list(["pl.scan_csv", "pl.scan_parquet"])
        stats = builder.get_stats()
        self.assertEqual(stats['total_built'], 2)
        self.assertEqual(stats['name'], "python_api_builder")


# ============================================================================
# Integration: PythonAPIBuilder + KnowledgeTree + Retrievers
# ============================================================================

class TestIntegrationWithRetrievers(unittest.TestCase):
    """Auto KTF 与 BM25Retriever 集成."""

    def test_auto_kt_searchable(self):
        """auto build 的 KT 可以被 retriever 检索."""
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars not installed")
        from knowledge_tree.python_api_builder import PythonAPIBuilder
        from knowledge_tree.core import KnowledgeTree
        from knowledge_tree.retrievers import BM25Retriever

        builder = PythonAPIBuilder()
        nodes = builder.build_from_api_list([
            "pl.scan_csv", "pl.scan_parquet", "DataFrame.pivot",
            "Expr.cum_sum", "Expr.str.contains",
        ])
        tree = KnowledgeTree(nodes)
        bm25 = BM25Retriever(tree)

        # 搜 "lazily read CSV" 应召回 scan_csv
        results = bm25.retrieve("Lazily read CSV file", top_k=2)
        ids = [n.id for n in results]
        self.assertIn("polars_scan_csv", ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
