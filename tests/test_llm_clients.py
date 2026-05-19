"""
tests/test_llm_clients.py
=========================

ClaudeCallable 单元测试. Mock anthropic SDK 避免真 API 调用.

测试覆盖:
  - 基础调用 + cost tracking
  - 重试逻辑 (rate limit / server error 重试; auth 不重试)
  - Budget 累计正确
  - Stats 接口
  - 异常分类 (retryable vs fatal)

运行:
  cd /home/claude && python -m unittest tests.test_llm_clients -v
"""

import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_mock_response(input_tokens=100, output_tokens=200, text="response"):
    """构造 mock anthropic.Message."""
    mock = MagicMock()
    mock.content = [MagicMock(text=text)]
    # 让 hasattr(b, "text") = True
    for b in mock.content:
        b.text = text
    mock.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return mock


def make_mock_api_error(status_code: int, message: str = "API error"):
    """构造 mock APIError."""
    from anthropic import APIError
    err = APIError.__new__(APIError)
    err.status_code = status_code
    err.message = message
    err.args = (message,)
    return err


class TestClaudeCallableBasic(unittest.TestCase):

    def setUp(self):
        # 跳过环境变量检查
        os.environ["ANTHROPIC_API_KEY"] = "test_key_dummy"

    def test_basic_call_succeeds(self):
        from knowledge_tree.llm_clients import ClaudeCallable

        client = ClaudeCallable(model="claude-sonnet-4-6", verbose=False)
        # Mock 内部 client.messages.create
        client.client.messages.create = MagicMock(
            return_value=make_mock_response(
                input_tokens=100, output_tokens=200, text="hello",
            )
        )

        result = client("test prompt")
        self.assertEqual(result, "hello")
        self.assertEqual(client.total_calls, 1)
        self.assertEqual(client.total_input_tokens, 100)
        self.assertEqual(client.total_output_tokens, 200)

    def test_cost_calculation_sonnet(self):
        """Sonnet 4.6: $3 input / $15 output per MTok."""
        from knowledge_tree.llm_clients import ClaudeCallable

        client = ClaudeCallable(model="claude-sonnet-4-6", verbose=False)
        client.client.messages.create = MagicMock(
            return_value=make_mock_response(
                input_tokens=1_000_000, output_tokens=1_000_000,
            )
        )
        client("test")
        # 1M input ($3) + 1M output ($15) = $18
        self.assertAlmostEqual(client.total_cost_usd, 18.0, places=4)

    def test_cost_calculation_haiku(self):
        """Haiku 4.5: $1 input / $5 output."""
        from knowledge_tree.llm_clients import ClaudeCallable

        client = ClaudeCallable(model="claude-haiku-4-5", verbose=False)
        client.client.messages.create = MagicMock(
            return_value=make_mock_response(
                input_tokens=1_000_000, output_tokens=1_000_000,
            )
        )
        client("test")
        self.assertAlmostEqual(client.total_cost_usd, 6.0, places=4)  # $1 + $5

    def test_multiple_calls_accumulate(self):
        from knowledge_tree.llm_clients import ClaudeCallable

        client = ClaudeCallable(verbose=False)
        client.client.messages.create = MagicMock(
            return_value=make_mock_response(100, 200)
        )
        client("p1")
        client("p2")
        client("p3")
        self.assertEqual(client.total_calls, 3)
        self.assertEqual(client.total_input_tokens, 300)
        self.assertEqual(client.total_output_tokens, 600)

    def test_empty_prompt_raises(self):
        from knowledge_tree.llm_clients import ClaudeCallable
        client = ClaudeCallable(verbose=False)
        with self.assertRaises(ValueError):
            client("")

    def test_unknown_model_warns(self):
        """未知模型 pricing 缺失, 应 warning 但不抛错."""
        from knowledge_tree.llm_clients import ClaudeCallable
        import logging
        with self.assertLogs(level=logging.WARNING) as cm:
            ClaudeCallable(model="claude-unknown-99", verbose=False)
        self.assertTrue(any("未知模型" in m for m in cm.output))

    def test_no_api_key_raises(self):
        from knowledge_tree.llm_clients import ClaudeCallable
        os.environ.pop("ANTHROPIC_API_KEY", None)
        with self.assertRaises(ValueError):
            ClaudeCallable(verbose=False)
        # 恢复
        os.environ["ANTHROPIC_API_KEY"] = "test_key_dummy"

    def test_stats_returns_dict(self):
        from knowledge_tree.llm_clients import ClaudeCallable
        client = ClaudeCallable(verbose=False)
        stats = client.get_stats()
        self.assertIn("model", stats)
        self.assertIn("total_calls", stats)
        self.assertIn("total_cost_usd", stats)

    def test_reset_stats(self):
        from knowledge_tree.llm_clients import ClaudeCallable
        client = ClaudeCallable(verbose=False)
        client.client.messages.create = MagicMock(
            return_value=make_mock_response(100, 200)
        )
        client("test")
        self.assertGreater(client.total_calls, 0)
        client.reset_stats()
        self.assertEqual(client.total_calls, 0)
        self.assertEqual(client.total_cost_usd, 0.0)


class TestClaudeCallableRetry(unittest.TestCase):

    def setUp(self):
        os.environ["ANTHROPIC_API_KEY"] = "test_key_dummy"

    def test_429_rate_limit_retries(self):
        """429 rate limit 应自动重试."""
        from knowledge_tree.llm_clients import ClaudeCallable
        from anthropic import APIError

        client = ClaudeCallable(
            max_retries=2, backoff_base_s=0.01, backoff_max_s=0.1,
            verbose=False,
        )

        call_count = [0]
        def mock_create(**kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                # 前 2 次 429
                raise make_mock_api_error(429, "rate limited")
            return make_mock_response(100, 200)

        client.client.messages.create = mock_create
        result = client("test")
        self.assertEqual(client.total_retries, 2)
        self.assertEqual(call_count[0], 3)

    def test_500_server_error_retries(self):
        from knowledge_tree.llm_clients import ClaudeCallable
        client = ClaudeCallable(
            max_retries=1, backoff_base_s=0.01, verbose=False,
        )
        call_count = [0]
        def mock_create(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise make_mock_api_error(503, "server error")
            return make_mock_response(100, 200)
        client.client.messages.create = mock_create

        client("test")
        self.assertEqual(call_count[0], 2)
        self.assertEqual(client.total_retries, 1)

    def test_401_auth_no_retry(self):
        """401 不重试, 直接 raise."""
        from knowledge_tree.llm_clients import ClaudeCallable, LLMFatalError

        client = ClaudeCallable(
            max_retries=3, backoff_base_s=0.01, verbose=False,
        )
        client.client.messages.create = MagicMock(
            side_effect=make_mock_api_error(401, "unauthorized")
        )
        with self.assertRaises(LLMFatalError):
            client("test")
        # 不应重试
        self.assertEqual(client.total_retries, 0)
        self.assertEqual(client.client.messages.create.call_count, 1)

    def test_404_model_not_found_no_retry(self):
        from knowledge_tree.llm_clients import ClaudeCallable, LLMFatalError
        client = ClaudeCallable(
            max_retries=3, backoff_base_s=0.01, verbose=False,
        )
        client.client.messages.create = MagicMock(
            side_effect=make_mock_api_error(404, "model not found")
        )
        with self.assertRaises(LLMFatalError):
            client("test")
        self.assertEqual(client.total_retries, 0)

    def test_max_retries_exceeded_raises_fatal(self):
        from knowledge_tree.llm_clients import ClaudeCallable, LLMFatalError
        client = ClaudeCallable(
            max_retries=2, backoff_base_s=0.01, verbose=False,
        )
        client.client.messages.create = MagicMock(
            side_effect=make_mock_api_error(429, "always throttled")
        )
        with self.assertRaises(LLMFatalError):
            client("test")
        self.assertEqual(client.total_retries, 2)

    def test_empty_content_retryable(self):
        """空 content 应触发重试."""
        from knowledge_tree.llm_clients import ClaudeCallable, LLMFatalError
        client = ClaudeCallable(
            max_retries=1, backoff_base_s=0.01, verbose=False,
        )

        empty_resp = MagicMock()
        empty_resp.content = []  # 空 content
        empty_resp.usage = MagicMock(input_tokens=0, output_tokens=0)

        good_resp = make_mock_response(100, 200, "hi")

        call_count = [0]
        def mock_create(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return empty_resp
            return good_resp
        client.client.messages.create = mock_create

        result = client("test")
        self.assertEqual(result, "hi")


class TestBudgetGuardedCallable(unittest.TestCase):

    def setUp(self):
        os.environ["ANTHROPIC_API_KEY"] = "test_key_dummy"

    def test_under_budget_passes(self):
        from knowledge_tree.llm_clients import ClaudeCallable
        # 直接 import 这俩
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts",
        ))
        from run_builders_real import (
            make_budget_guarded_callable, BudgetExceeded,
        )

        inner = ClaudeCallable(verbose=False)
        inner.client.messages.create = MagicMock(
            return_value=make_mock_response(100, 200)
        )
        guarded = make_budget_guarded_callable(inner, max_cost_usd=1.0)
        # 第 1 次: cost ~ 0.003003 (100 * $3/1M + 200 * $15/1M)
        # cost = 0.0003 + 0.003 = 0.0033, 远小于 $1
        result = guarded("test")
        self.assertIsNotNone(result)

    def test_over_budget_raises(self):
        from knowledge_tree.llm_clients import ClaudeCallable
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts",
        ))
        from run_builders_real import (
            make_budget_guarded_callable, BudgetExceeded,
        )

        inner = ClaudeCallable(verbose=False)
        # 模拟已花了 $5
        inner.total_cost_usd = 5.0
        inner.client.messages.create = MagicMock(
            return_value=make_mock_response(100, 200)
        )
        guarded = make_budget_guarded_callable(inner, max_cost_usd=1.0)
        with self.assertRaises(BudgetExceeded):
            guarded("test")

    def test_stat_passthrough(self):
        from knowledge_tree.llm_clients import ClaudeCallable
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts",
        ))
        from run_builders_real import make_budget_guarded_callable

        inner = ClaudeCallable(verbose=False)
        guarded = make_budget_guarded_callable(inner, max_cost_usd=100.0)
        # 应能透传 total_cost_usd 等属性
        self.assertEqual(guarded.total_cost_usd, 0.0)
        self.assertEqual(guarded.total_calls, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
