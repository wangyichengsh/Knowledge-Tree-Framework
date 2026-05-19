"""
tests/test_aime_dryrun.py
=========================

aime_evaluator_dryrun.py 单元测试.

测试覆盖:
  - run_evaluator_dryrun 在合法 evaluator 上不报 Bug D
  - run_evaluator_dryrun 在故障 evaluator 上检出 Bug D
  - AIME_OUTPUT_FORMATS 覆盖关键场景
  - load_aime_2024 字段映射 (mock datasets)

运行:
  cd /home/claude && python -m unittest tests.test_aime_dryrun -v
"""

import os
import sys
import unittest

# 路径配置
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
))

from aime_evaluator_dryrun import (
    AIME_OUTPUT_FORMATS,
    run_evaluator_dryrun,
)


# ============================================================================
# Mock evaluator (合法 - 模拟 Tool 1 v4 的行为)
# ============================================================================

def mock_extract_good(text: str) -> str:
    """模拟合法 evaluator: 提取 \\boxed{...} 内容, 支持嵌套 brace.

    实现 balanced bracket parser, 模拟 phase2_mcts.extract_boxed_answer 行为.
    """
    # 找所有 \boxed{ 起始位置
    results = []
    i = 0
    while True:
        start = text.find(r"\boxed{", i)
        if start < 0:
            break
        # 从 { 开始 balanced bracket
        brace_start = start + len(r"\boxed{")
        depth = 1
        j = brace_start
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            # j 是 } 的下一个位置
            content = text[brace_start : j - 1]
            results.append(content)
            i = j
        else:
            break  # 未闭合, 停

    if not results:
        return ""

    last = results[-1].strip()
    # 处理 \text{} wrap (re-apply balanced bracket)
    if last.startswith(r"\text{") and last.endswith("}"):
        return last[len(r"\text{") : -1].strip()
    return last


def mock_match_good(a: str, b: str) -> bool:
    """模拟合法 match: 字符串相等 + 数值等价 + LaTeX frac 转换."""
    import re

    if not a or not b:
        return False
    if a == b:
        return True

    def _normalize(s: str) -> str:
        """简化: \\frac{a}{b} -> a/b, 去空白."""
        s = s.strip()
        # \frac{num}{den} -> num/den
        s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", s)
        return s

    a_n = _normalize(a)
    b_n = _normalize(b)

    if a_n == b_n:
        return True

    try:
        # Float 比较 (处理 leading zero)
        return abs(float(a_n) - float(b_n)) < 1e-6
    except ValueError:
        pass

    # Fraction "a/b" 比较
    def _try_frac(s):
        if "/" in s:
            try:
                num, den = s.split("/", 1)
                return float(num) / float(den)
            except (ValueError, ZeroDivisionError):
                return None
        try:
            return float(s)
        except ValueError:
            return None

    a_val = _try_frac(a_n)
    b_val = _try_frac(b_n)
    if a_val is not None and b_val is not None:
        return abs(a_val - b_val) < 1e-6

    return False


# ============================================================================
# Mock evaluator (有 Bug - 不处理 leading zero)
# ============================================================================

def mock_extract_buggy(text: str) -> str:
    """与 mock_extract_good 相同."""
    return mock_extract_good(text)


def mock_match_buggy(a: str, b: str) -> bool:
    """有 Bug: 只做字符串相等比较, 不处理 leading zero / fraction."""
    if not a or not b:
        return False
    return a == b


# ============================================================================
# 测试
# ============================================================================

class TestEvaluatorDryrun(unittest.TestCase):

    def test_good_evaluator_zero_bug_d(self):
        """合法 evaluator 应在所有 cases 上零 Bug D."""
        results, bug_d = run_evaluator_dryrun(
            mock_extract_good, mock_match_good, target_gt="35",
        )
        self.assertEqual(len(results), len(AIME_OUTPUT_FORMATS))
        self.assertEqual(
            len(bug_d), 0,
            f"合法 evaluator 不应有 Bug D, 但发现: {bug_d}",
        )

    def test_buggy_evaluator_detects_bug_d(self):
        """有 Bug 的 evaluator (不处理 leading zero / fraction) 应被检出."""
        results, bug_d = run_evaluator_dryrun(
            mock_extract_buggy, mock_match_buggy, target_gt="35",
        )
        # 应该检出 leading_zero / fraction 类 cases 是 Bug D
        self.assertGreater(len(bug_d), 0)
        self.assertIn("leading_zero_3", bug_d)
        self.assertIn("fraction_redundant", bug_d)

    def test_output_format_keys_complete(self):
        """每个结果都有完整字段."""
        results, _ = run_evaluator_dryrun(
            mock_extract_good, mock_match_good, target_gt="35",
        )
        for r in results:
            self.assertIn("case", r)
            self.assertIn("input_text", r)
            self.assertIn("expected_pred", r)
            self.assertIn("actual_pred", r)
            self.assertIn("expected_match", r)
            self.assertIn("actual_match", r)
            self.assertIn("bug_d_candidate", r)
            self.assertIn("note", r)

    def test_aime_output_formats_completeness(self):
        """AIME_OUTPUT_FORMATS 覆盖关键场景."""
        labels = [c[0] for c in AIME_OUTPUT_FORMATS]
        # 必须含的关键场景
        required = [
            "standard_boxed",       # 基线
            "leading_zero_3",        # AIME 整数答案常见
            "fraction_redundant",    # 分数化简
            "wrong_answer",          # 合法格式错答 (false positive 检测)
            "truncated_no_boxed",    # 截断 (max_new_tokens 满)
        ]
        for r in required:
            self.assertIn(r, labels, f"缺少关键场景: {r}")

    def test_extract_exception_caught(self):
        """extract 抛错时应捕获为 Bug D 候选, 不传播."""
        def crashing_extract(text):
            raise RuntimeError("simulated extract crash")

        results, bug_d = run_evaluator_dryrun(
            crashing_extract, mock_match_good, target_gt="35",
        )
        # 所有 cases 都该被标 Bug D (因为 extract 全 crash)
        self.assertEqual(len(bug_d), len(AIME_OUTPUT_FORMATS))
        for r in results:
            self.assertTrue(r["bug_d_candidate"])
            self.assertIn("ERROR", r["actual_pred"])

    def test_match_exception_caught(self):
        """match 抛错时应捕获."""
        def crashing_match(a, b):
            raise RuntimeError("simulated match crash")

        results, bug_d = run_evaluator_dryrun(
            mock_extract_good, crashing_match, target_gt="35",
        )
        # 任何 extract 成功的 case 都会触发 match crash
        non_empty_count = sum(1 for r in results if r["actual_pred"])
        self.assertGreater(non_empty_count, 0)
        # 至少一些 case 被标 Bug D (match 失败)
        for r in results:
            if r["actual_pred"]:
                self.assertTrue(r["bug_d_candidate"])


# ============================================================================
# AIME 数据加载 (mock datasets 库)
# ============================================================================

class TestLoadAime(unittest.TestCase):

    def test_load_with_huggingfaceh4_format(self):
        """HuggingFaceH4 字段小写 (problem/answer/id)."""
        import sys
        # Mock datasets.load_dataset
        mock_data = [
            {"id": 0, "problem": "Q1", "solution": "...", "answer": "35", "url": "", "year": "2024"},
            {"id": 1, "problem": "Q2", "solution": "...", "answer": "120", "url": "", "year": "2024"},
        ]

        import types
        fake_datasets = types.ModuleType("datasets")
        fake_datasets.load_dataset = lambda name, split: mock_data
        sys.modules["datasets"] = fake_datasets

        # 重新 import (清缓存)
        if "aime_evaluator_dryrun" in sys.modules:
            del sys.modules["aime_evaluator_dryrun"]
        from aime_evaluator_dryrun import load_aime_2024

        samples = load_aime_2024("HuggingFaceH4/aime_2024")
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0]["question"], "Q1")
        self.assertEqual(samples[0]["gt_answer"], "35")
        self.assertEqual(samples[0]["aime_id"], "0")
        self.assertEqual(samples[0]["level"], "AIME")

    def test_load_with_maxwell_jia_format(self):
        """Maxwell-Jia 字段 PascalCase (Problem/Answer/ID)."""
        import sys, types

        mock_data = [
            {"ID": "2024-I-1", "Problem": "Q1", "Solution": "...", "Answer": "204"},
        ]
        fake_datasets = types.ModuleType("datasets")
        fake_datasets.load_dataset = lambda name, split: mock_data
        sys.modules["datasets"] = fake_datasets

        if "aime_evaluator_dryrun" in sys.modules:
            del sys.modules["aime_evaluator_dryrun"]
        from aime_evaluator_dryrun import load_aime_2024

        samples = load_aime_2024("Maxwell-Jia/AIME_2024")
        self.assertEqual(samples[0]["question"], "Q1")
        self.assertEqual(samples[0]["gt_answer"], "204")
        self.assertEqual(samples[0]["aime_id"], "2024-I-1")

    def test_load_warns_on_non_integer_answer(self):
        """非整数 GT 应 warning (但不抛错, 容错)."""
        import sys, types, logging

        mock_data = [
            {"id": 0, "problem": "Q1", "solution": "...", "answer": "not_a_number"},
        ]
        fake_datasets = types.ModuleType("datasets")
        fake_datasets.load_dataset = lambda name, split: mock_data
        sys.modules["datasets"] = fake_datasets

        if "aime_evaluator_dryrun" in sys.modules:
            del sys.modules["aime_evaluator_dryrun"]
        from aime_evaluator_dryrun import load_aime_2024

        with self.assertLogs(level=logging.WARNING) as cm:
            samples = load_aime_2024()
        self.assertEqual(len(samples), 1)
        self.assertTrue(any("不是整数" in msg for msg in cm.output))

    def test_load_warns_on_out_of_range(self):
        """超出 [0, 999] 范围应 warning."""
        import sys, types, logging

        mock_data = [
            {"id": 0, "problem": "Q1", "solution": "...", "answer": "1500"},
        ]
        fake_datasets = types.ModuleType("datasets")
        fake_datasets.load_dataset = lambda name, split: mock_data
        sys.modules["datasets"] = fake_datasets

        if "aime_evaluator_dryrun" in sys.modules:
            del sys.modules["aime_evaluator_dryrun"]
        from aime_evaluator_dryrun import load_aime_2024

        with self.assertLogs(level=logging.WARNING) as cm:
            samples = load_aime_2024()
        self.assertTrue(any("超出" in msg for msg in cm.output))


if __name__ == "__main__":
    unittest.main(verbosity=2)
