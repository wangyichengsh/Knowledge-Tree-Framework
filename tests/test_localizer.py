"""tests/test_localizer.py — 两阶段定位测试."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge_tree.localizer import (
    localize, reorder_by_localization, _parse_localization_response,
    _build_candidates_listing, LocalizationResult,
)
from knowledge_tree.core import KnowledgeNode


def _node(nid, qn, defn="some function", sig="def f(self)", file="m.py"):
    return KnowledgeNode(
        id=nid, title=qn.split('.')[-1], definition=defn,
        source_code=f"{sig}: pass",
        domain_metadata={'qualified_name': qn, 'signature': sig, 'file': file,
                         'type': 'method' if '.' in qn else 'function'},
    )


class TestParseLocalizationResponse(unittest.TestCase):
    def test_clean_json(self):
        resp = '{"selected_ids": ["a", "b"], "reasoning": "because"}'
        ids, reason = _parse_localization_response(resp, {"a", "b", "c"})
        self.assertEqual(ids, ["a", "b"])
        self.assertEqual(reason, "because")

    def test_markdown_fenced(self):
        resp = '```json\n{"selected_ids": ["a"], "reasoning": "x"}\n```'
        ids, reason = _parse_localization_response(resp, {"a"})
        self.assertEqual(ids, ["a"])

    def test_filters_hallucinated_ids(self):
        """LLM 选了不存在的 id → 过滤掉 (PROTO-7.37)."""
        resp = '{"selected_ids": ["a", "ghost", "b"]}'
        ids, _ = _parse_localization_response(resp, {"a", "b"})
        self.assertEqual(ids, ["a", "b"])  # ghost 被过滤

    def test_no_json(self):
        ids, reason = _parse_localization_response("no json here", {"a"})
        self.assertEqual(ids, [])

    def test_extra_prose_around_json(self):
        resp = 'Sure! Here:\n{"selected_ids": ["a"]}\nHope that helps'
        ids, _ = _parse_localization_response(resp, {"a"})
        self.assertEqual(ids, ["a"])

    def test_ordinal_fallback_bare_number(self):
        """LLM 返回裸序号 '1' '13' (1-based) → 映射回真实 id (Day 9 实证)."""
        ordered = [f"node_{i}" for i in range(15)]  # node_0 .. node_14
        valid = set(ordered)
        resp = '{"selected_ids": ["1", "13"]}'
        ids, _ = _parse_localization_response(resp, valid, ordered)
        # "1" → ordered[0] = node_0, "13" → ordered[12] = node_12
        self.assertEqual(ids, ["node_0", "node_12"])

    def test_ordinal_fallback_id_prefix(self):
        """LLM 返回 'id1' 'id15' → 映射回真实 id (Day 9 实证)."""
        ordered = [f"node_{i}" for i in range(15)]
        valid = set(ordered)
        resp = '{"selected_ids": ["id1", "id15"]}'
        ids, _ = _parse_localization_response(resp, valid, ordered)
        # id1 → ordered[0], id15 → ordered[14]
        self.assertEqual(ids, ["node_0", "node_14"])

    def test_ordinal_out_of_range_skipped(self):
        """序号超范围 (如 99) → skip, 不崩."""
        ordered = ["a", "b", "c"]
        resp = '{"selected_ids": ["99", "2"]}'
        ids, _ = _parse_localization_response(resp, set(ordered), ordered)
        # 99 超范围 skip, 2 → ordered[1] = b
        self.assertEqual(ids, ["b"])

    def test_real_id_preferred_over_ordinal(self):
        """真实 id 优先 (不误触发序号映射)."""
        ordered = ["alpha", "beta", "5"]  # 注意 id 本身是 "5"
        resp = '{"selected_ids": ["5"]}'
        ids, _ = _parse_localization_response(resp, set(ordered), ordered)
        # "5" 是真实 id (在 valid_ids), 优先匹配它, 而非序号映射到 ordered[4]
        self.assertEqual(ids, ["5"])

    def test_no_ordered_ids_no_fallback(self):
        """不传 ordered_ids → 序号无法映射, skip (向后兼容)."""
        resp = '{"selected_ids": ["1", "valid_id"]}'
        ids, _ = _parse_localization_response(resp, {"valid_id"})
        self.assertEqual(ids, ["valid_id"])


class TestLocalize(unittest.TestCase):
    def test_few_candidates_skip_llm(self):
        """candidates <= select_k → 全选, 不调 LLM."""
        cands = [_node("n1", "A.foo"), _node("n2", "A.bar")]
        called = []
        llm = lambda p: called.append(p) or '{"selected_ids": []}'
        result = localize("bug", cands, llm, select_k=3)
        self.assertEqual(set(result.selected_ids), {"n1", "n2"})
        self.assertEqual(called, [])  # LLM 没被调用

    def test_llm_selects_subset(self):
        cands = [_node(f"n{i}", f"A.m{i}") for i in range(10)]
        llm = lambda p: '{"selected_ids": ["n3", "n7"], "reasoning": "these hold the bug"}'
        result = localize("bug report", cands, llm, select_k=3)
        self.assertEqual(result.selected_ids, ["n3", "n7"])
        self.assertEqual(result.reasoning, "these hold the bug")
        self.assertFalse(result.fell_back)

    def test_llm_failure_fallback(self):
        cands = [_node(f"n{i}", f"A.m{i}") for i in range(10)]
        def bad_llm(p):
            raise RuntimeError("model crashed")
        result = localize("bug", cands, bad_llm, select_k=3)
        self.assertTrue(result.fell_back)
        self.assertEqual(result.selected_ids, ["n0", "n1", "n2"])  # top-3 fallback

    def test_llm_empty_selection_fallback(self):
        cands = [_node(f"n{i}", f"A.m{i}") for i in range(10)]
        llm = lambda p: '{"selected_ids": []}'
        result = localize("bug", cands, llm, select_k=3)
        self.assertTrue(result.fell_back)
        self.assertEqual(result.selected_ids, ["n0", "n1", "n2"])

    def test_select_k_truncation(self):
        cands = [_node(f"n{i}", f"A.m{i}") for i in range(10)]
        llm = lambda p: '{"selected_ids": ["n1", "n2", "n3", "n4", "n5"]}'
        result = localize("bug", cands, llm, select_k=3)
        self.assertEqual(len(result.selected_ids), 3)


class TestReorder(unittest.TestCase):
    def test_selected_first(self):
        cands = [_node(f"n{i}", f"A.m{i}") for i in range(5)]
        loc = LocalizationResult(selected_ids=["n3", "n1"])
        reordered = reorder_by_localization(cands, loc)
        # 选中的在前 (LLM 顺序), 其余补后
        self.assertEqual(reordered[0].id, "n3")
        self.assertEqual(reordered[1].id, "n1")
        # 其余 n0, n2, n4 在后
        rest_ids = [n.id for n in reordered[2:]]
        self.assertEqual(set(rest_ids), {"n0", "n2", "n4"})

    def test_all_candidates_preserved(self):
        cands = [_node(f"n{i}", f"A.m{i}") for i in range(5)]
        loc = LocalizationResult(selected_ids=["n2"])
        reordered = reorder_by_localization(cands, loc)
        self.assertEqual(len(reordered), 5)  # 没丢节点


class TestCandidatesListing(unittest.TestCase):
    def test_listing_contains_key_info(self):
        cands = [_node("n1", "SQLCompiler.get_order_by", sig="def get_order_by(self)")]
        listing = _build_candidates_listing(cands)
        self.assertIn("n1", listing)
        self.assertIn("SQLCompiler.get_order_by", listing)
        self.assertIn("get_order_by", listing)

    def test_listing_no_full_source(self):
        """listing 不含完整 source_code (省 token, 那是 Stage 2 的事)."""
        cands = [_node("n1", "A.foo")]
        cands[0].source_code = "def foo():\n    " + "x = 1\n    " * 100  # 长 source
        listing = _build_candidates_listing(cands)
        # listing 应远短于 source_code
        self.assertLess(len(listing), len(cands[0].source_code))


if __name__ == "__main__":
    unittest.main(verbosity=2)
