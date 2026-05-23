"""tests/test_claude_api_client.py — Claude API callable (mock, 不真调 API)."""

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge_tree.claude_api_client import ClaudeAPICallable


def _mock_response(text, input_tokens=100, output_tokens=50):
    """构造 Anthropic API 响应 JSON."""
    return json.dumps({
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }).encode("utf-8")


class _FakeHTTPResponse:
    def __init__(self, body):
        self._body = body
    def read(self):
        return self._body
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


class TestClaudeAPICallable(unittest.TestCase):
    def test_requires_api_key(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                ClaudeAPICallable(api_key=None)

    def test_basic_call_extracts_text(self):
        c = ClaudeAPICallable(api_key="test-key")
        with mock.patch("urllib.request.urlopen",
                        return_value=_FakeHTTPResponse(_mock_response("Hello response"))):
            result = c("test prompt")
        self.assertEqual(result, "Hello response")
        self.assertEqual(c.get_stats()["calls"], 1)
        self.assertEqual(c.get_stats()["input_tokens"], 100)

    def test_skips_thinking_blocks(self):
        """thinking block 应被跳过, 只返回 text block."""
        body = json.dumps({
            "content": [
                {"type": "thinking", "thinking": "internal reasoning"},
                {"type": "text", "text": "actual answer"},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }).encode("utf-8")
        c = ClaudeAPICallable(api_key="test-key")
        with mock.patch("urllib.request.urlopen",
                        return_value=_FakeHTTPResponse(body)):
            result = c("prompt")
        self.assertEqual(result, "actual answer")
        self.assertNotIn("internal reasoning", result)

    def test_multiple_text_blocks_joined(self):
        body = json.dumps({
            "content": [
                {"type": "text", "text": "part1"},
                {"type": "text", "text": "part2"},
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }).encode("utf-8")
        c = ClaudeAPICallable(api_key="test-key")
        with mock.patch("urllib.request.urlopen",
                        return_value=_FakeHTTPResponse(body)):
            result = c("prompt")
        self.assertEqual(result, "part1\npart2")

    def test_thinking_budget_sets_body(self):
        """thinking_budget>0 时, 请求体含 thinking 字段 + temperature=1 (非-adaptive 模型)."""
        c = ClaudeAPICallable(model="claude-sonnet-4-6", api_key="test-key", thinking_budget=2000)
        captured = {}
        def fake_urlopen(req, timeout=None):
            captured['data'] = json.loads(req.data.decode("utf-8"))
            return _FakeHTTPResponse(_mock_response("ok"))
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            c("prompt")
        self.assertIn("thinking", captured['data'])
        self.assertEqual(captured['data']['thinking']['budget_tokens'], 2000)
        self.assertEqual(captured['data']['temperature'], 1.0)

    def test_unload_is_noop(self):
        c = ClaudeAPICallable(api_key="test-key")
        c.unload()  # 不应崩

    def test_retry_on_429(self):
        """429 应触发重试."""
        import urllib.error
        c = ClaudeAPICallable(api_key="test-key", max_retries=2)
        call_count = [0]
        def fake_urlopen(req, timeout=None):
            call_count[0] += 1
            if call_count[0] == 1:
                raise urllib.error.HTTPError(
                    "url", 429, "rate limited", {}, None)
            return _FakeHTTPResponse(_mock_response("recovered"))
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with mock.patch("time.sleep"):  # 跳过真 sleep
                result = c("prompt")
        self.assertEqual(result, "recovered")
        self.assertEqual(call_count[0], 2)

    def test_non_retryable_4xx_raises(self):
        """400 (非 429) 不重试, 直接 raise."""
        import urllib.error
        c = ClaudeAPICallable(api_key="test-key", max_retries=3)
        def fake_urlopen(req, timeout=None):
            raise urllib.error.HTTPError("url", 400, "bad request", {}, None)
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(RuntimeError):
                c("prompt")


class TestAdaptiveAdaptation(unittest.TestCase):
    """Opus 4.7+ adaptive-thinking-only 自动适配 (不传 temperature/budget_tokens)."""

    def _capture_body(self, c, prompt="p"):
        captured = {}
        def fake_urlopen(req, timeout=None):
            captured['data'] = json.loads(req.data.decode("utf-8"))
            return _FakeHTTPResponse(_mock_response("ok"))
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            c(prompt)
        return captured['data']

    def test_opus_47_detected_adaptive(self):
        from knowledge_tree.claude_api_client import _is_adaptive_thinking_only
        self.assertTrue(_is_adaptive_thinking_only("claude-opus-4-7"))
        self.assertFalse(_is_adaptive_thinking_only("claude-sonnet-4-6"))
        self.assertFalse(_is_adaptive_thinking_only("claude-opus-4-6"))

    def test_opus_47_no_temperature(self):
        """Opus 4.7: body 不含 temperature (API 会拒绝)."""
        c = ClaudeAPICallable(model="claude-opus-4-7", api_key="test-key")
        body = self._capture_body(c)
        self.assertNotIn("temperature", body)
        self.assertNotIn("thinking", body)

    def test_opus_47_no_budget_tokens_even_if_set(self):
        """Opus 4.7: 即使设了 thinking_budget 也不传 (adaptive only)."""
        c = ClaudeAPICallable(model="claude-opus-4-7", api_key="test-key",
                              thinking_budget=5000)
        body = self._capture_body(c)
        self.assertNotIn("thinking", body)

    def test_opus_47_effort_passed(self):
        """Opus 4.7: effort 档位被传入."""
        c = ClaudeAPICallable(model="claude-opus-4-7", api_key="test-key",
                              effort="high")
        body = self._capture_body(c)
        self.assertEqual(body.get("effort"), "high")

    def test_sonnet_46_keeps_temperature(self):
        """Sonnet 4.6: 保留 temperature."""
        c = ClaudeAPICallable(model="claude-sonnet-4-6", api_key="test-key",
                              temperature=0.7)
        body = self._capture_body(c)
        self.assertEqual(body.get("temperature"), 0.7)

    def test_sonnet_46_thinking_budget_works(self):
        """Sonnet 4.6: thinking_budget 走 enabled thinking."""
        c = ClaudeAPICallable(model="claude-sonnet-4-6", api_key="test-key",
                              thinking_budget=3000)
        body = self._capture_body(c)
        self.assertIn("thinking", body)
        self.assertEqual(body["thinking"]["budget_tokens"], 3000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
