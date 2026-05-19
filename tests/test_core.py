"""
tests/test_core.py
==================

KnowledgeNode + KnowledgeTree + WorkedExample 单元测试.

测试策略 (PROTO-7.9 单元测试 + 实数据 dual-validation):
  - 字段验证 (空字段 raise / 越界值 raise)
  - 序列化往返 (to_dict -> from_dict 等价)
  - 树结构操作 (add / get / get_children / get_descendants)
  - validate() 检测各类不一致
  - bm25_index_text 不含 worked_examples (用户决策 D-2)
  - llm_inject_text 含 worked_examples (T-3.7)

运行:
  cd /home/claude && python -m pytest tests/test_core.py -v
  或: python -m unittest tests.test_core -v
"""

import unittest
from knowledge_tree.core import (
    KnowledgeNode,
    KnowledgeTree,
    WorkedExample,
)


# ============================================================================
# WorkedExample 测试
# ============================================================================

class TestWorkedExample(unittest.TestCase):

    def test_create_minimal(self):
        ex = WorkedExample(
            problem="Compute C(5, 2)",
            solution_steps=["Apply formula C(n,k) = n!/(k!(n-k)!)", "C(5,2) = 10"],
            final_answer="10",
        )
        self.assertEqual(ex.final_answer, "10")
        self.assertEqual(ex.key_insight, "")  # 默认空

    def test_empty_problem_raises(self):
        with self.assertRaises(ValueError):
            WorkedExample(problem="", solution_steps=["a"], final_answer="1")

    def test_empty_solution_steps_raises(self):
        with self.assertRaises(ValueError):
            WorkedExample(problem="p", solution_steps=[], final_answer="1")

    def test_empty_final_answer_raises(self):
        with self.assertRaises(ValueError):
            WorkedExample(problem="p", solution_steps=["a"], final_answer="")

    def test_serialize_roundtrip(self):
        ex = WorkedExample(
            problem="Test",
            solution_steps=["s1", "s2"],
            final_answer="42",
            key_insight="Use Vieta",
        )
        d = ex.to_dict()
        ex2 = WorkedExample.from_dict(d)
        self.assertEqual(ex.problem, ex2.problem)
        self.assertEqual(ex.solution_steps, ex2.solution_steps)
        self.assertEqual(ex.final_answer, ex2.final_answer)
        self.assertEqual(ex.key_insight, ex2.key_insight)


# ============================================================================
# KnowledgeNode 测试
# ============================================================================

class TestKnowledgeNode(unittest.TestCase):

    def _make_node(self, node_id="n1", **kwargs) -> KnowledgeNode:
        defaults = {
            "id": node_id,
            "title": "Test Node",
            "definition": "A test concept.",
        }
        defaults.update(kwargs)
        return KnowledgeNode(**defaults)

    def test_create_minimal(self):
        node = self._make_node()
        self.assertEqual(node.id, "n1")
        self.assertEqual(node.confidence, 1.0)
        self.assertEqual(node.source, "manual")
        self.assertEqual(node.worked_examples, [])

    def test_empty_id_raises(self):
        with self.assertRaises(ValueError):
            KnowledgeNode(id="", title="t", definition="d")
        with self.assertRaises(ValueError):
            KnowledgeNode(id="   ", title="t", definition="d")

    def test_empty_title_raises(self):
        with self.assertRaises(ValueError):
            KnowledgeNode(id="n1", title="", definition="d")

    def test_empty_definition_raises(self):
        with self.assertRaises(ValueError):
            KnowledgeNode(id="n1", title="t", definition="")

    def test_confidence_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            self._make_node(confidence=-0.1)
        with self.assertRaises(ValueError):
            self._make_node(confidence=1.1)

    def test_bm25_index_text_excludes_worked_examples(self):
        """用户决策 D-2: worked_examples 不进 BM25 索引."""
        ex = WorkedExample(
            problem="UNIQUE_PROBLEM_TEXT_ZZZZ",
            solution_steps=["UNIQUE_STEP_TEXT_YYYY"],
            final_answer="UNIQUE_ANSWER_XXXX",
        )
        node = self._make_node(
            title="Combinatorics",
            definition="Counting things.",
            key_facts=["Binomial: C(n,k) = n!/(k!(n-k)!)"],
            worked_examples=[ex],
            related_concepts=["Permutation"],
        )
        index_text = node.bm25_index_text()

        # 进索引的字段 ✅
        self.assertIn("Combinatorics", index_text)
        self.assertIn("Counting things", index_text)
        self.assertIn("Binomial", index_text)
        self.assertIn("Permutation", index_text)

        # 不进索引的字段 ❌ (worked_examples 内容)
        self.assertNotIn("UNIQUE_PROBLEM_TEXT_ZZZZ", index_text)
        self.assertNotIn("UNIQUE_STEP_TEXT_YYYY", index_text)
        self.assertNotIn("UNIQUE_ANSWER_XXXX", index_text)

    def test_llm_inject_text_includes_worked_examples(self):
        """T-3.7: LLM inject 必须含 worked_examples."""
        ex = WorkedExample(
            problem="Compute C(5, 2)",
            solution_steps=["Apply formula", "Get 10"],
            final_answer="10",
            key_insight="Standard binomial",
        )
        node = self._make_node(
            title="Binomial",
            worked_examples=[ex],
            common_pitfalls=["Don't confuse with permutation"],
        )
        text = node.llm_inject_text()
        self.assertIn("Compute C(5, 2)", text)
        self.assertIn("Standard binomial", text)
        self.assertIn("Don't confuse with permutation", text)
        self.assertIn("### Worked Examples", text)
        self.assertIn("### Common Pitfalls", text)

    def test_serialize_roundtrip_with_worked_examples(self):
        """T-3.7 + 工程: 含嵌套 dataclass 的序列化必须双向无损."""
        ex = WorkedExample(
            problem="P1",
            solution_steps=["s1"],
            final_answer="A1",
            key_insight="K1",
        )
        node = self._make_node(
            id="combo",
            title="Combo",
            definition="...",
            key_facts=["fact1", "fact2"],
            worked_examples=[ex],
            common_pitfalls=["pit1"],
            children_ids=["c1", "c2"],
            related_concepts=["r1"],
            confidence=0.9,
            source="wikipedia",
            domain_metadata={"latex_form": "$C(n,k)$"},
        )
        d = node.to_dict()
        node2 = KnowledgeNode.from_dict(d)

        self.assertEqual(node.id, node2.id)
        self.assertEqual(node.title, node2.title)
        self.assertEqual(node.confidence, node2.confidence)
        self.assertEqual(len(node2.worked_examples), 1)
        self.assertIsInstance(node2.worked_examples[0], WorkedExample)
        self.assertEqual(node2.worked_examples[0].problem, "P1")
        self.assertEqual(node2.domain_metadata, {"latex_form": "$C(n,k)$"})

    def test_from_dict_handles_missing_optional_fields(self):
        """向后兼容: 旧版 JSON 缺少 v2 字段时, 用默认值."""
        minimal = {
            "id": "n1",
            "title": "T",
            "definition": "D",
        }
        node = KnowledgeNode.from_dict(minimal)
        self.assertEqual(node.confidence, 1.0)
        self.assertEqual(node.source, "manual")
        self.assertEqual(node.domain_metadata, {})


# ============================================================================
# KnowledgeTree 测试
# ============================================================================

class TestKnowledgeTree(unittest.TestCase):

    def _make_tree_3_nodes(self) -> KnowledgeTree:
        """构造一个 3 节点树: root -> [child_a, child_b]."""
        root = KnowledgeNode(
            id="root", title="Root", definition="Root concept",
            children_ids=["a", "b"],
        )
        a = KnowledgeNode(
            id="a", title="A", definition="A concept", parent_id="root",
        )
        b = KnowledgeNode(
            id="b", title="B", definition="B concept", parent_id="root",
        )
        return KnowledgeTree(nodes=[root, a, b])

    def test_basic_add_get(self):
        tree = KnowledgeTree()
        n = KnowledgeNode(id="n1", title="N", definition="D")
        tree.add_node(n)
        self.assertEqual(len(tree), 1)
        self.assertTrue(tree.has_node("n1"))
        self.assertEqual(tree.get_node("n1").title, "N")

    def test_duplicate_id_raises(self):
        tree = KnowledgeTree()
        n1 = KnowledgeNode(id="n1", title="A", definition="D")
        n2 = KnowledgeNode(id="n1", title="B", definition="D")
        tree.add_node(n1)
        with self.assertRaises(ValueError):
            tree.add_node(n2)  # 重复 id

    def test_get_node_missing_raises(self):
        tree = KnowledgeTree()
        with self.assertRaises(KeyError):
            tree.get_node("nonexistent")

    def test_get_children(self):
        tree = self._make_tree_3_nodes()
        children = tree.get_children("root")
        ids = sorted(c.id for c in children)
        self.assertEqual(ids, ["a", "b"])

    def test_get_parent(self):
        tree = self._make_tree_3_nodes()
        self.assertEqual(tree.get_parent("a").id, "root")
        self.assertIsNone(tree.get_parent("root"))

    def test_get_root_ids(self):
        tree = self._make_tree_3_nodes()
        roots = tree.get_root_ids()
        self.assertEqual(roots, ["root"])

    def test_get_root_ids_multiple_roots(self):
        """KTF 允许 forest (多 root)."""
        n1 = KnowledgeNode(id="r1", title="R1", definition="d")
        n2 = KnowledgeNode(id="r2", title="R2", definition="d")
        tree = KnowledgeTree(nodes=[n1, n2])
        roots = tree.get_root_ids()
        self.assertEqual(roots, ["r1", "r2"])

    def test_get_descendants(self):
        """root -> [a, b], a -> [a1]."""
        a1 = KnowledgeNode(id="a1", title="A1", definition="d", parent_id="a")
        a = KnowledgeNode(
            id="a", title="A", definition="d", parent_id="root",
            children_ids=["a1"],
        )
        b = KnowledgeNode(id="b", title="B", definition="d", parent_id="root")
        root = KnowledgeNode(
            id="root", title="R", definition="d", children_ids=["a", "b"],
        )
        tree = KnowledgeTree(nodes=[root, a, b, a1])

        # 全后代
        descendants = tree.get_descendants("root")
        ids = sorted(d.id for d in descendants)
        self.assertEqual(ids, ["a", "a1", "b"])

        # max_depth=1
        descendants = tree.get_descendants("root", max_depth=1)
        ids = sorted(d.id for d in descendants)
        self.assertEqual(ids, ["a", "b"])

    def test_validate_clean_tree(self):
        tree = self._make_tree_3_nodes()
        issues = tree.validate(strict=False)
        self.assertEqual(issues, [])

    def test_validate_orphan_child_id(self):
        """children_ids 引用不存在的 id."""
        n = KnowledgeNode(
            id="n1", title="N", definition="D",
            children_ids=["nonexistent"],
        )
        tree = KnowledgeTree(nodes=[n])
        issues = tree.validate(strict=False)
        self.assertTrue(any("nonexistent" in iss for iss in issues))

    def test_validate_orphan_parent_id(self):
        """parent_id 引用不存在的 id."""
        n = KnowledgeNode(
            id="n1", title="N", definition="D",
            parent_id="ghost",
        )
        tree = KnowledgeTree(nodes=[n])
        issues = tree.validate(strict=False)
        self.assertTrue(any("ghost" in iss for iss in issues))

    def test_validate_bidirectional_inconsistency(self):
        """A.parent_id=B 但 B.children_ids 不含 A."""
        a = KnowledgeNode(id="a", title="A", definition="d", parent_id="b")
        b = KnowledgeNode(id="b", title="B", definition="d")  # children_ids 不含 'a'
        tree = KnowledgeTree(nodes=[a, b])
        issues = tree.validate(strict=False)
        self.assertTrue(any("双向不一致" in iss for iss in issues))

    def test_validate_strict_raises(self):
        n = KnowledgeNode(
            id="n1", title="N", definition="D",
            children_ids=["nonexistent"],
        )
        tree = KnowledgeTree(nodes=[n])
        with self.assertRaises(ValueError):
            tree.validate(strict=True)

    def test_validate_multi_parent_conflict(self):
        """KTF v2 是树, 不是 DAG. 多个父声明同一子节点应报错."""
        a = KnowledgeNode(
            id="a", title="A", definition="d",
            children_ids=["shared"],
        )
        b = KnowledgeNode(
            id="b", title="B", definition="d",
            children_ids=["shared"],
        )
        # shared 被 a 和 b 同时声明为 child
        shared = KnowledgeNode(id="shared", title="S", definition="d", parent_id="a")
        tree = KnowledgeTree(nodes=[a, b, shared])
        issues = tree.validate(strict=False)
        self.assertTrue(any("多父冲突" in iss for iss in issues))

    def test_get_descendants_dedup_on_dag(self):
        """即使数据是 DAG (多父), get_descendants 也不重复返回."""
        a = KnowledgeNode(
            id="a", title="A", definition="d",
            children_ids=["x", "y"],
        )
        x = KnowledgeNode(
            id="x", title="X", definition="d",
            children_ids=["y"],  # x 也声明 y 是子, 形成 DAG (a -> x -> y, a -> y)
        )
        y = KnowledgeNode(id="y", title="Y", definition="d")
        tree = KnowledgeTree(nodes=[a, x, y])

        descendants = tree.get_descendants("a")
        ids = [d.id for d in descendants]
        # y 应只出现一次, 不能因 DAG 重复
        self.assertEqual(ids.count("y"), 1)
        self.assertEqual(sorted(ids), ["x", "y"])

    def test_stats(self):
        ex = WorkedExample(
            problem="P", solution_steps=["s"], final_answer="A",
        )
        n1 = KnowledgeNode(id="n1", title="T1", definition="D", worked_examples=[ex])
        n2 = KnowledgeNode(id="n2", title="T2", definition="D", source="wikipedia")
        tree = KnowledgeTree(nodes=[n1, n2])

        stats = tree.stats()
        self.assertEqual(stats["n_nodes"], 2)
        self.assertEqual(stats["n_with_worked_examples"], 1)
        self.assertEqual(stats["sources"], {"manual": 1, "wikipedia": 1})


if __name__ == "__main__":
    unittest.main(verbosity=2)
