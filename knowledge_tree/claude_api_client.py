"""
knowledge_tree/claude_api_client.py
====================================

Phase 4.3 Day 9: Claude API 作为 generator/localizer callable.

动机:
  R1 同时做 retrieval-localize + generation, 能力噪音大, 无法区分
  "localization 机制本身有害" vs "R1 能力拖累了机制".
  用 Claude API 解耦: 强模型跑同一 pipeline, 测 KTF 框架能力上限.

接口与 LocalModelCallable 一致:
  - __call__(prompt: str) -> str
  - unload()  (no-op, API 无需卸载)

设计:
  - 纯 stdlib (urllib), 不依赖 anthropic SDK (用户环境不用额外装包)
  - 从环境变量 ANTHROPIC_API_KEY 读 key
  - 重试 + 超时处理
  - 可选 thinking (extended thinking 对 localization/复杂 bug 有帮助)
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


def _is_adaptive_thinking_only(model: str) -> bool:
    """判断模型是否为 adaptive-thinking-only (移除了 sampling 参数 + budget_tokens).

    Opus 4.7 起 (claude-opus-4-7): adaptive thinking only, 不接受 temperature /
    top_p / budget_tokens, 改用 effort 参数. 传这些参数会 API 报错.

    其他模型 (Sonnet 4.6, Opus 4.6, Haiku 4.5 等): 保留传统 temperature +
    enabled-thinking (budget_tokens).

    检测规则: 模型名匹配 opus-4-7 或更高. 保守起见只硬匹配已知的 adaptive-only 串.
    """
    m = model.lower()
    # opus-4-7 及之后的 opus (4-8, 4-9...) 都是 adaptive-only
    if "opus-4-7" in m or "opus-4-8" in m or "opus-4-9" in m:
        return True
    # 未来 5.x 也大概率 adaptive-only, 但不预判, 用户可显式控制
    return False



class ClaudeAPICallable:
    """Claude API callable, 接口兼容 LocalModelCallable.

    用法:
        model = ClaudeAPICallable(model="claude-opus-4-20250514")
        response = model("your prompt")
        model.unload()  # no-op
    """

    def __init__(
        self,
        model: str = "claude-opus-4-7",
        max_tokens: int = 8192,
        temperature: float = 1.0,
        api_key: str | None = None,
        max_retries: int = 3,
        timeout: int = 300,
        thinking_budget: int = 0,  # >0 启用 extended thinking (仅非-adaptive 模型)
        effort: str | None = None,  # adaptive 模型的努力档位: low/medium/high/xhigh
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set (env var or api_key arg)")
        self.max_retries = max_retries
        self.timeout = timeout
        self.thinking_budget = thinking_budget
        self.effort = effort
        self.verbose = verbose
        # 自动检测: adaptive-thinking-only 模型 (Opus 4.7+)
        self.adaptive_only = _is_adaptive_thinking_only(model)
        if self.adaptive_only and verbose:
            logger.info("Model %s is adaptive-thinking-only: "
                        "ignoring temperature/budget_tokens, using effort=%s",
                        model, effort or "default")
        self._call_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def __call__(self, prompt: str) -> str:
        """发 prompt 给 Claude, 返回文本响应 (不含 thinking blocks)."""
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        if self.adaptive_only:
            # Opus 4.7+: adaptive thinking only, 不传 temperature / budget_tokens.
            # 可选 effort 档位 (low/medium/high/xhigh) 控制推理深度.
            if self.effort:
                body["effort"] = self.effort
            # 不设 temperature (API 会拒绝)
        else:
            # 传统模型 (Sonnet 4.6 / Opus 4.6 / Haiku 4.5): temperature + enabled thinking
            if self.thinking_budget > 0:
                body["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget}
                body["temperature"] = 1.0  # enabled thinking 要求 temperature=1
            else:
                body["temperature"] = self.temperature

        data = json.dumps(body).encode("utf-8")
        headers = {
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }

        last_err = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    ANTHROPIC_API_URL, data=data, headers=headers, method="POST"
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8")
                result = json.loads(raw)

                # 抽 text blocks (跳过 thinking blocks)
                text_parts = []
                for block in result.get("content", []):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                text = "\n".join(text_parts)

                # 统计
                self._call_count += 1
                usage = result.get("usage", {})
                self._total_input_tokens += usage.get("input_tokens", 0)
                self._total_output_tokens += usage.get("output_tokens", 0)
                if self.verbose:
                    logger.info(
                        "Claude API call #%d: in=%d out=%d tokens",
                        self._call_count, usage.get("input_tokens", 0),
                        usage.get("output_tokens", 0),
                    )
                return text

            except urllib.error.HTTPError as e:
                err_body = ""
                try:
                    err_body = e.read().decode("utf-8")
                except Exception:
                    pass
                last_err = f"HTTP {e.code}: {err_body[:300]}"
                # 429 / 529 (overloaded) / 5xx → 重试 with backoff
                if e.code in (429, 500, 502, 503, 529):
                    wait = 2 ** attempt * 5
                    logger.warning("Claude API %s, retry in %ds (attempt %d/%d)",
                                   last_err, wait, attempt + 1, self.max_retries)
                    time.sleep(wait)
                    continue
                else:
                    # 4xx (非 429) 不重试
                    logger.error("Claude API non-retryable error: %s", last_err)
                    raise RuntimeError(f"Claude API error: {last_err}")
            except (urllib.error.URLError, TimeoutError) as e:
                last_err = str(e)
                wait = 2 ** attempt * 5
                logger.warning("Claude API network error %s, retry in %ds (attempt %d/%d)",
                               last_err, wait, attempt + 1, self.max_retries)
                time.sleep(wait)
                continue

        raise RuntimeError(f"Claude API failed after {self.max_retries} retries: {last_err}")

    def unload(self) -> None:
        """No-op (API 无需卸载). 打印用量统计."""
        logger.info(
            "ClaudeAPICallable stats: %d calls, %d input tokens, %d output tokens",
            self._call_count, self._total_input_tokens, self._total_output_tokens,
        )

    def get_stats(self) -> dict:
        return {
            "calls": self._call_count,
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
        }
