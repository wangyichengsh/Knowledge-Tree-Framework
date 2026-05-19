"""
tests/test_fix_tree_errors.py
==============================

fix_tree_errors.py 单元测试.

测试覆盖:
  - format_other_examples (跳过 bad example, 格式化其他)
  - fix_one_example 各种 success / failure 路径
  - 修复后 example 实际写入 node
  - verify 失败时不应该崩, 但 audit 记录失败
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

from fix_tree_errors import fix_one_example, format_other_examples
from knowledge_tree.core import KnowledgeNode, WorkedExample


def make_test_node(num_examples=3) -> KnowledgeNode:
    examples = []
    for i in range(num_examples):
        examples.append(WorkedExample(
            problem=f"Problem {i}: do something with params X={i}",
            solution_steps=[f"step {i}.1", f"step {i}.2"],
            final_answer=f"answer{i}",
            key_insight=f"insight {i}",
        ))
    return KnowledgeNode(
        id="test_node",
        title="Test Concept",
        definition="A test concept for unit testing.",
        key_facts=["fact 1", "fact 2"],
        worked_examples=examples,
    )


def make_test_error(example_index=2):
    """1-indexed example_index (与 review report 一致)."""
    return {
        'id': 'E_TEST',
        'node_id': 'test_node',
        'example_index': example_index,
        'severity': 'A',
        'type': 'numerical_error',
        'summary': 'Test error summary',
        'verification': 'Test verification',
        'recommendation': 'Test recommendation',
    }


# ============================================================================
# format_other_examples
# ============================================================================

class TestFormatOtherExamples(unittest.TestCase):

    def test_skips_target_example(self):
        """target example 不应在输出中."""
        node = make_test_node(num_examples=3)
        # skip_index=1 (即 example #2)
        result = format_other_examples(node, skip_index=1)
        self.assertIn("Problem 0", result)
        self.assertNotIn("Problem 1", result)
        self.assertIn("Problem 2", result)

    def test_returns_placeholder_when_only_one_example(self):
        """如果节点只有 1 个 example, 排除它后无其他."""
        node = make_test_node(num_examples=1)
        result = format_other_examples(node, skip_index=0)
        self.assertIn("no other examples", result.lower())

    def test_skip_index_out_of_range(self):
        """skip_index 超出范围: 全部 examples 都展示."""
        node = make_test_node(num_examples=2)
        result = format_other_examples(node, skip_index=99)
        self.assertIn("Problem 0", result)
        self.assertIn("Problem 1", result)


# ============================================================================
# fix_one_example
# ============================================================================

class TestFixOneExample(unittest.TestCase):

    def test_success_path_with_verify(self):
        """正常路径: LLM 返回 valid JSON, verify 通过."""
        responses = [
            # 1st call: generate
            json.dumps({
                "problem": "Fixed problem with params 5, 3",
                "solution_steps": ["correct step 1", "correct step 2"],
                "final_answer": "15",
                "key_insight": "Apply the correct formula",
            }),
            # 2nd call: verify
            json.dumps({"verified": True, "confidence": 0.95}),
        ]
        mock = MagicMock(side_effect=responses)

        node = make_test_node(num_examples=3)
        error = make_test_error(example_index=2)

        new_ex, audit = fix_one_example(mock, node, error, verify=True)

        self.assertIsNotNone(new_ex)
        self.assertEqual(new_ex.final_answer, "15")
        self.assertIn("Fixed problem", new_ex.problem)
        self.assertEqual(audit['status'], 'generated')
        self.assertTrue(audit['verify_result']['verified'])
        # 2 calls: gen + verify
        self.assertEqual(mock.call_count, 2)

    def test_success_path_no_verify(self):
        """跳过 verify: 只调一次 LLM."""
        mock = MagicMock(return_value=json.dumps({
            "problem": "p", "solution_steps": ["s"],
            "final_answer": "a", "key_insight": "k",
        }))

        node = make_test_node()
        error = make_test_error()
        new_ex, audit = fix_one_example(mock, node, error, verify=False)
        self.assertIsNotNone(new_ex)
        self.assertEqual(mock.call_count, 1)
        self.assertNotIn('verify_result', audit)

    def test_invalid_example_index(self):
        """example_index 越界: 应返回 None + failed_invalid_index."""
        node = make_test_node(num_examples=3)
        error = make_test_error(example_index=99)  # 越界

        mock = MagicMock()
        new_ex, audit = fix_one_example(mock, node, error, verify=False)
        self.assertIsNone(new_ex)
        self.assertEqual(audit['status'], 'failed_invalid_index')
        # 不应调 LLM
        self.assertEqual(mock.call_count, 0)

    def test_llm_call_failure(self):
        """LLM 抛异常: 返回 None + failed_llm_call."""
        mock = MagicMock(side_effect=RuntimeError("LLM down"))

        node = make_test_node()
        error = make_test_error()
        new_ex, audit = fix_one_example(mock, node, error, verify=False)
        self.assertIsNone(new_ex)
        self.assertEqual(audit['status'], 'failed_llm_call')
        self.assertIn("LLM down", audit['reason'])

    def test_invalid_json_response(self):
        """LLM 返回非 JSON: 返回 None + failed_json_parse."""
        mock = MagicMock(return_value="this is not json at all")

        node = make_test_node()
        error = make_test_error()
        new_ex, audit = fix_one_example(mock, node, error, verify=False)
        self.assertIsNone(new_ex)
        self.assertEqual(audit['status'], 'failed_json_parse')

    def test_missing_required_field(self):
        """LLM 返回缺字段 JSON: 返回 None + failed_missing_fields."""
        mock = MagicMock(return_value=json.dumps({
            "problem": "p",
            # missing solution_steps, final_answer, key_insight
        }))

        node = make_test_node()
        error = make_test_error()
        new_ex, audit = fix_one_example(mock, node, error, verify=False)
        self.assertIsNone(new_ex)
        self.assertEqual(audit['status'], 'failed_missing_fields')
        self.assertIn('solution_steps', audit['missing'])

    def test_verify_failure_still_returns_example(self):
        """Verify 失败时, 仍返回 example (用户决定要不要用), 但 audit 记 status."""
        responses = [
            # gen
            json.dumps({
                "problem": "p",
                "solution_steps": ["s"],
                "final_answer": "a",
                "key_insight": "k",
            }),
            # verify fails
            json.dumps({"verified": False, "issues": ["bad step 2"]}),
        ]
        mock = MagicMock(side_effect=responses)

        node = make_test_node()
        error = make_test_error()
        new_ex, audit = fix_one_example(mock, node, error, verify=True)

        # example 仍生成 (verify 不阻塞)
        self.assertIsNotNone(new_ex)
        # audit 标记 verify 失败
        self.assertEqual(audit['status'], 'generated_but_verify_failed')
        self.assertFalse(audit['verify_result']['verified'])
        self.assertIn('bad step 2', audit['verify_result']['issues'])

    def test_audit_contains_original_data(self):
        """audit 应保留原 problem / answer 便于审计."""
        mock = MagicMock(side_effect=[
            json.dumps({"problem": "new_p", "solution_steps": ["s"],
                        "final_answer": "new_a", "key_insight": "new_k"}),
            json.dumps({"verified": True}),
        ])

        node = make_test_node(num_examples=3)
        error = make_test_error(example_index=2)  # 1-indexed → idx 1
        new_ex, audit = fix_one_example(mock, node, error, verify=True)

        # 原 example idx=1 是 "Problem 1: do something with params X=1"
        self.assertIn("Problem 1", audit['original_problem'])
        self.assertEqual(audit['original_answer'], "answer1")
        self.assertEqual(audit['new_answer'], "new_a")


if __name__ == "__main__":
    unittest.main(verbosity=2)
