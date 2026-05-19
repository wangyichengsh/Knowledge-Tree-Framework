"""
tests/test_fix_related_concepts.py
====================================

fix_related_concepts.py 单元测试.

测试覆盖:
  - fuzzy_match 基础和边界
  - fix_node Stage 1 only
  - fix_node Stage 1 + Stage 2 mock LLM
  - LLM judge 解析容错
  - Self-reference 排除

运行:
  cd /home/claude && python -m unittest tests.test_fix_related_concepts -v
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
))

from fix_related_concepts import (
    fuzzy_match,
    llm_judge_mapping,
    fix_node,
)
from knowledge_tree.core import KnowledgeNode, WorkedExample


# ============================================================================
# Fuzzy match
# ============================================================================

class TestFuzzyMatch(unittest.TestCase):

    def test_substring_match(self):
        """'polynomial_roots' 应映射到 'polynomial_roots_and_coefficients'."""
        available = [
            "polynomial_roots_and_coefficients",
            "factoring_techniques",
            "vietas_formulas",
        ]
        result = fuzzy_match("polynomial_roots", available)
        self.assertEqual(result[0], "polynomial_roots_and_coefficients")

    def test_close_string_match(self):
        """'polynom_root' (typo-like) 应能 fuzzy match."""
        available = [
            "polynomial_roots_and_coefficients",
            "modular_arithmetic",
        ]
        result = fuzzy_match("polynomial_root", available, cutoff=0.5)
        # 应能匹配到含 'polynomial' 的
        self.assertTrue(any("polynomial" in r for r in result))

    def test_no_match(self):
        """完全无关的 ref 应返回空."""
        available = ["combinatorics", "geometry", "number_theory"]
        result = fuzzy_match("xyz_completely_unrelated", available)
        self.assertEqual(result, [])

    def test_empty_inputs(self):
        self.assertEqual(fuzzy_match("", ["a", "b"]), [])
        self.assertEqual(fuzzy_match("x", []), [])

    def test_max_matches_respected(self):
        # 多个相似项, 应限制返回数量
        available = [
            "polynomial_roots_and_coefficients",
            "polynomial_division_and_remainder_theorem",
            "polynomial_factoring_techniques",
            "polynomial_long_division",
        ]
        result = fuzzy_match("polynomial", available, cutoff=0.3, max_matches=2)
        self.assertLessEqual(len(result), 2)

    def test_exact_match(self):
        available = ["polynomial_roots", "other"]
        result = fuzzy_match("polynomial_roots", available)
        self.assertEqual(result[0], "polynomial_roots")


# ============================================================================
# fix_node (Stage 1 only)
# ============================================================================

class TestFixNodeStage1(unittest.TestCase):

    def _make_node(self, related: list[str]) -> KnowledgeNode:
        ex = WorkedExample(
            problem="p", solution_steps=["s"],
            final_answer="a", key_insight="k",
        )
        return KnowledgeNode(
            id="test_node",
            title="Test",
            definition="A test concept.",
            key_facts=["fact 1", "fact 2"],
            worked_examples=[ex],
            related_concepts=related,
        )

    def test_no_broken_refs_no_change(self):
        """所有 refs 已 valid: 不变."""
        available = ["concept_a", "concept_b", "test_node"]
        node = self._make_node(["concept_a", "concept_b"])

        fixed, stats = fix_node(node, available, stage1_only=True)
        self.assertEqual(stats["broken_count"], 0)
        self.assertEqual(stats["stage1_resolved"], 0)
        self.assertEqual(sorted(fixed.related_concepts), ["concept_a", "concept_b"])

    def test_broken_ref_fuzzy_resolved(self):
        """broken ref 通过 fuzzy match 解决."""
        available = [
            "polynomial_roots_and_coefficients",  # 期待匹配目标
            "factoring_techniques",
            "test_node",
        ]
        node = self._make_node([
            "polynomial_roots",  # broken, 应映射到 polynomial_roots_and_coefficients
            "factoring_techniques",  # OK
        ])

        fixed, stats = fix_node(node, available, stage1_only=True)
        self.assertEqual(stats["broken_count"], 1)
        self.assertEqual(stats["stage1_resolved"], 1)
        self.assertIn("polynomial_roots_and_coefficients", fixed.related_concepts)
        self.assertIn("factoring_techniques", fixed.related_concepts)
        self.assertNotIn("polynomial_roots", fixed.related_concepts)  # 原 broken 被替换

    def test_broken_ref_no_fuzzy_match_stage1_only_rejected(self):
        """Stage 1 only 模式: 无 fuzzy match 的 broken refs 被丢弃."""
        available = ["combinatorics", "geometry", "test_node"]
        node = self._make_node([
            "completely_unrelated_xyz",  # 无任何 fuzzy 匹配
        ])

        fixed, stats = fix_node(node, available, stage1_only=True)
        self.assertEqual(stats["broken_count"], 1)
        self.assertEqual(stats["stage1_resolved"], 0)
        self.assertEqual(stats["stage2_rejected"], 1)  # Stage 1 only 模式下丢弃
        self.assertNotIn("completely_unrelated_xyz", fixed.related_concepts)

    def test_self_reference_excluded(self):
        """自引用应被排除 (即使在 available_ids 中)."""
        # 用合理长度的 ids 避免 fuzzy_match 误匹配
        available = ["test_node", "another_concept_id"]
        node = self._make_node(["test_node", "another_concept_id"])
        # 'test_node' 是 node.id, 即使在 available 也应排除
        # 'another_concept_id' 在 available, 应保留

        fixed, stats = fix_node(node, available, stage1_only=True)
        # 'test_node' (自引用) 被排除
        # 'another_concept_id' 保留
        self.assertEqual(fixed.related_concepts, ["another_concept_id"])

    def test_audit_metadata_preserved(self):
        """原 broken refs 应记在 domain_metadata."""
        available = ["polynomial_roots_and_coefficients", "test_node"]
        node = self._make_node(["polynomial_roots", "made_up_id"])

        fixed, stats = fix_node(node, available, stage1_only=True)
        self.assertIn("original_related_refs_broken", fixed.domain_metadata)
        self.assertIn(
            "polynomial_roots",
            fixed.domain_metadata["original_related_refs_broken"],
        )
        self.assertIn("fix_stage1_resolutions", fixed.domain_metadata)


# ============================================================================
# fix_node (Stage 1 + Stage 2 LLM)
# ============================================================================

class TestFixNodeStage2(unittest.TestCase):

    def _make_node(self, related: list[str]) -> KnowledgeNode:
        ex = WorkedExample(
            problem="p", solution_steps=["s"],
            final_answer="a", key_insight="k",
        )
        return KnowledgeNode(
            id="test_node", title="Test",
            definition="Test concept",
            key_facts=["f1", "f2"],
            worked_examples=[ex],
            related_concepts=related,
        )

    def test_llm_resolves_when_fuzzy_fails(self):
        """fuzzy 找不到时, LLM judge 解决."""
        available = ["category_x_advanced", "category_y_basic", "test_node"]
        node = self._make_node(["something_completely_different"])

        # Mock LLM: 返回 'category_x_advanced' 作为映射
        mock_llm = MagicMock(
            return_value='{"mappings": {"something_completely_different": "category_x_advanced"}}'
        )

        fixed, stats = fix_node(
            node, available, callable_=mock_llm, stage1_only=False,
        )

        # Stage 1 应该没解决 (fuzzy match 找不到)
        # Stage 2 LLM 解决
        self.assertEqual(stats["stage1_resolved"], 0)
        self.assertEqual(stats["stage2_resolved"], 1)
        self.assertIn("category_x_advanced", fixed.related_concepts)

    def test_llm_rejects_when_no_good_match(self):
        """LLM 判定无 match (返回 null) 应被记录."""
        available = ["a", "b", "test_node"]
        node = self._make_node(["unrelated_xyz"])

        mock_llm = MagicMock(
            return_value='{"mappings": {"unrelated_xyz": null}}'
        )
        fixed, stats = fix_node(
            node, available, callable_=mock_llm, stage1_only=False,
        )
        self.assertEqual(stats["stage2_rejected"], 1)
        self.assertNotIn("unrelated_xyz", fixed.related_concepts)

    def test_llm_invalid_response_treated_as_rejection(self):
        """LLM 响应格式错: 不应崩溃, 视为全 reject."""
        available = ["a", "b", "test_node"]
        node = self._make_node(["xyz"])

        mock_llm = MagicMock(return_value="not valid json")
        fixed, stats = fix_node(
            node, available, callable_=mock_llm, stage1_only=False,
        )
        self.assertEqual(stats["stage2_rejected"], 1)
        self.assertNotIn("xyz", fixed.related_concepts)

    def test_llm_maps_to_invalid_id_rejected(self):
        """LLM 返回的映射不在白名单: 应被拒绝."""
        available = ["a", "b", "test_node"]
        node = self._make_node(["xyz"])

        # LLM 返回 'ghost_id' (不在 available)
        mock_llm = MagicMock(
            return_value='{"mappings": {"xyz": "ghost_id"}}'
        )
        fixed, stats = fix_node(
            node, available, callable_=mock_llm, stage1_only=False,
        )
        # 应被识别为非法, rejected
        self.assertEqual(stats["stage2_rejected"], 1)
        self.assertNotIn("ghost_id", fixed.related_concepts)

    def test_no_duplicates_in_final(self):
        """fuzzy match + valid 已含同样 id: 不重复."""
        available = ["polynomial_roots_and_coefficients", "test_node"]
        node = self._make_node([
            "polynomial_roots_and_coefficients",  # 已 valid
            "polynomial_roots",  # 应 fuzzy 映射到同样的
        ])

        fixed, stats = fix_node(node, available, stage1_only=True)
        # 只应出现一次
        self.assertEqual(
            fixed.related_concepts.count("polynomial_roots_and_coefficients"),
            1,
        )


# ============================================================================
# 集成: 真实 hierarchy + 真实 broken refs (模拟你的 tree.json 场景)
# ============================================================================

class TestRealisticScenario(unittest.TestCase):

    def test_phase_4_1_week_2_scenario(self):
        """
        模拟你 Phase 4.1 Week 2 实测发现的场景.

        节点 vietas_formulas 含 related_concepts:
          ['polynomial_roots', 'symmetric_polynomials', 'quadratic_formula',
           'polynomial_factoring', 'newton_power_sums', 'complex_roots_conjugate_pairs']
        
        Hierarchy 中实际有:
          polynomial_roots_and_coefficients, symmetric_functions, 
          factoring_techniques, vietas_formulas, ...
        """
        available = [
            "polynomial_roots_and_coefficients",
            "symmetric_functions",
            "factoring_techniques",
            "vietas_formulas",
            "absolute_value_equations_and_inequalities",
            "complex_numbers_algebra",
        ]

        # 构造 vietas_formulas 节点 (来自你真实 tree.json)
        ex = WorkedExample(
            problem="Find p^2+q^2+r^2", solution_steps=["s1"],
            final_answer="21", key_insight="vieta",
        )
        node = KnowledgeNode(
            id="vietas_formulas",
            title="Vieta's Formulas",
            definition="Express coefficients as symmetric functions of roots.",
            key_facts=["fact 1", "fact 2"],
            worked_examples=[ex],
            related_concepts=[
                "polynomial_roots",  # 应 fuzzy → polynomial_roots_and_coefficients
                "symmetric_polynomials",  # 应 fuzzy → symmetric_functions
                "quadratic_formula",  # 可能无 fuzzy match
                "polynomial_factoring",  # 应 fuzzy → factoring_techniques
                "newton_power_sums",  # 无 match
                "complex_roots_conjugate_pairs",  # 应 fuzzy → complex_numbers_algebra
            ],
        )

        fixed, stats = fix_node(node, available, stage1_only=True)

        # 至少修复了几个
        self.assertGreaterEqual(stats["stage1_resolved"], 2)
        # 修复后所有 related_concepts 都在白名单
        for r in fixed.related_concepts:
            self.assertIn(r, available, f"{r} 不在白名单")
        # 不应有自引用
        self.assertNotIn("vietas_formulas", fixed.related_concepts)


if __name__ == "__main__":
    unittest.main(verbosity=2)
