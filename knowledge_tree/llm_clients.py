"""
knowledge_tree/llm_clients.py
==============================

LLM API 包装. 提供 LLMCallable 接口的具体实现 (Claude / 本地 / mock).

框架依赖:
  framework v3.4 PROTO-7.16 (借理念不依赖工具): LLMCallable 抽象隔离 SDK
                                                  本模块是 anthropic SDK 唯一接口点
  framework v3.4 用户决策: builder LLM = Claude API

设计原则:
  (1) 单一职责: 这里只做 LLM 调用 + rate limit + cost tracking
      不做 prompt 设计 (builder 职责) / 不做响应解析 (caller 职责)

  (2) 失败处理: 区分可重试 (rate limit / 网络) vs 不可重试 (auth / 模型 ID 错)
      可重试: exponential backoff (1s, 2s, 4s, ...)
      不可重试: 立即 raise

  (3) Cost tracking:
      每次调用记录 input_tokens / output_tokens (从 response.usage)
      累计 total_cost (按模型 pricing 算)
      用户可随时查看当前花费

  (4) 模型选择:
      默认 claude-sonnet-4-6 (best price-performance for Phase 4.1 概念建树)
      可选 claude-haiku-4-5 (3x 便宜, 但 worked_examples 质量略低)
      Phase 4.1 用 Sonnet, Phase 4.2 大 corpus 时考虑 Haiku

成本估算 (基于 framework v3.4 + 实测 Tool 3 v2 经验):
  Phase 4.1 builder 用量: 300-500 概念 × ~1.5K input + ~1.5K output tokens
  Sonnet 4.6 cost: 300 × ($3 × 1.5/1000 + $15 × 1.5/1000) ≈ $8.10
  含 retry buffer: ~$12 总预算

用法:
  from knowledge_tree.llm_clients import ClaudeCallable
  
  client = ClaudeCallable(
      api_key=os.environ["ANTHROPIC_API_KEY"],
      model="claude-sonnet-4-6",
      max_tokens=2048,
  )
  
  response = client("Tell me about binomial coefficients.")
  print(f"Used: {client.total_input_tokens} input, {client.total_output_tokens} output")
  print(f"Cost so far: ${client.total_cost_usd:.4f}")
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# 模型 pricing (2026-05 实测, USD per million tokens)
# ============================================================================

# 数据源: web search 2026-05-11, pricepertoken.com + finout.io
# 验证: docs.claude.com/en/api/overview (生产时可补)
MODEL_PRICING = {
    # 当前默认
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-opus-4-7": {"input": 5.00, "output": 25.00},
    "claude-opus-4-6": {"input": 5.00, "output": 25.00},
    # 旧版 (备用)
    "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
}

DEFAULT_MODEL = "claude-sonnet-4-6"


# ============================================================================
# 异常类型 (区分可重试 vs 不可重试)
# ============================================================================

class LLMRetryableError(Exception):
    """可重试错误: rate limit, transient network, server 5xx"""


class LLMFatalError(Exception):
    """不可重试错误: auth 401, model 不存在 404, malformed request"""


# ============================================================================
# ClaudeCallable - 主类
# ============================================================================

class ClaudeCallable:
    """
    Claude API 的 LLMCallable 实现.

    实例本身是 callable: `client(prompt)` 返回 response text.

    Attributes (实时统计):
        total_calls: 累计调用次数 (含失败重试)
        total_input_tokens: 累计 input tokens
        total_output_tokens: 累计 output tokens
        total_cost_usd: 累计花费 (USD)
        total_retries: 累计重试次数

    线程安全: 否. Phase 4.1 串行使用即可.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        max_retries: int = 4,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 60.0,
        rate_limit_calls_per_minute: Optional[int] = None,
        cost_alert_threshold_usd: Optional[float] = None,
        verbose: bool = True,
    ) -> None:
        """
        Args:
            api_key: Anthropic API key (默认从 ANTHROPIC_API_KEY 环境变量读)
            model: 模型 ID (默认 claude-sonnet-4-6)
            max_tokens: 单次 generation 最大输出 (默认 2048)
            temperature: 采样温度 (默认 0.7, builder 用稍高鼓励多样性)
            max_retries: 重试次数 (默认 4)
            backoff_base_s: 重试基础等待秒数 (默认 1, exponential: 1s, 2s, 4s, 8s)
            backoff_max_s: 重试最大等待 (默认 60)
            rate_limit_calls_per_minute: 客户端限速 (None 表示不限, 依赖 API 端)
            cost_alert_threshold_usd: 累计花费超阈值打 warning (None 表示不限)
            verbose: 每次调用 log 进度
        """
        # 延迟 import anthropic (允许 mock 测试时不需要)
        try:
            from anthropic import Anthropic, APIError
            self._Anthropic = Anthropic
            self._APIError = APIError
        except ImportError as e:
            raise RuntimeError(
                "anthropic SDK 未安装. 运行: pip install anthropic --break-system-packages"
            ) from e

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key 未提供. 设置 ANTHROPIC_API_KEY 环境变量或传 api_key 参数"
            )

        if model not in MODEL_PRICING:
            logger.warning(
                "未知模型 %r, pricing 未配置, cost tracking 不准. "
                "已知模型: %s",
                model, list(MODEL_PRICING.keys()),
            )

        self.client = self._Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self.rate_limit_calls_per_minute = rate_limit_calls_per_minute
        self.cost_alert_threshold_usd = cost_alert_threshold_usd
        self.verbose = verbose

        # 统计
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.total_retries = 0

        # Rate limit 跟踪 (sliding window)
        self._recent_call_times: list[float] = []

    def __call__(self, prompt: str) -> str:
        """
        LLMCallable 协议: prompt -> response.

        失败处理:
          - LLMRetryableError: 按 exponential backoff 重试
          - LLMFatalError: 立即抛错
          - 超过 max_retries: 抛 LLMFatalError
        """
        if not prompt:
            raise ValueError("prompt 不能为空")

        # Rate limit: 客户端 sliding window
        if self.rate_limit_calls_per_minute is not None:
            self._enforce_rate_limit()

        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self._call_once(prompt)
                self._recent_call_times.append(time.time())
                return response
            except LLMFatalError:
                raise
            except LLMRetryableError as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = min(
                        self.backoff_base_s * (2 ** attempt),
                        self.backoff_max_s,
                    )
                    if self.verbose:
                        logger.warning(
                            "ClaudeCallable: 重试 %d/%d (等待 %.1fs), 错误: %s",
                            attempt + 1, self.max_retries, delay, e,
                        )
                    self.total_retries += 1
                    time.sleep(delay)
                    continue
                # 超过 max_retries
                raise LLMFatalError(
                    f"超过 max_retries={self.max_retries}, 最后错误: {e}"
                )

        # 不应到达
        raise LLMFatalError(f"未知重试退出, last_error={last_error}")

    def _call_once(self, prompt: str) -> str:
        """
        单次 API 调用. 根据异常类型分类为 retryable / fatal.

        Returns:
            response text

        Raises:
            LLMRetryableError: 可重试错误
            LLMFatalError: 不可重试错误
        """
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
        except self._APIError as e:
            status = getattr(e, "status_code", None)
            # 分类
            if status == 401 or status == 403:
                raise LLMFatalError(f"Auth 失败 (status {status}): {e}")
            if status == 404:
                raise LLMFatalError(f"模型不存在或路径错 (status 404): {e}")
            if status == 400:
                raise LLMFatalError(f"请求格式错 (status 400): {e}")
            if status == 429:
                raise LLMRetryableError(f"Rate limit (429): {e}")
            if status and 500 <= status < 600:
                raise LLMRetryableError(f"Server error ({status}): {e}")
            # 未知 status: 谨慎判定可重试
            raise LLMRetryableError(f"未知 API error (status {status}): {e}")
        except Exception as e:
            # 网络层错误 (httpx.ConnectError 等)
            msg = str(e).lower()
            if any(k in msg for k in ["timeout", "connection", "network"]):
                raise LLMRetryableError(f"Network error: {e}")
            raise LLMFatalError(f"Unexpected error: {type(e).__name__}: {e}")

        # 提取 text
        if not message.content:
            raise LLMRetryableError("API 返回空 content (可能是 throttle 或 internal)")

        # message.content 是 list[ContentBlock], 取第一个 text block
        text_blocks = [
            b.text for b in message.content
            if hasattr(b, "text") and b.text
        ]
        if not text_blocks:
            raise LLMRetryableError(f"响应无 text content: {message.content}")
        response_text = "\n".join(text_blocks)

        # 更新统计
        self.total_calls += 1
        usage = getattr(message, "usage", None)
        if usage:
            in_tok = getattr(usage, "input_tokens", 0)
            out_tok = getattr(usage, "output_tokens", 0)
            self.total_input_tokens += in_tok
            self.total_output_tokens += out_tok
            # Cost
            pricing = MODEL_PRICING.get(self.model)
            if pricing:
                this_cost = (
                    in_tok * pricing["input"] / 1_000_000
                    + out_tok * pricing["output"] / 1_000_000
                )
                self.total_cost_usd += this_cost

                # Alert
                if (self.cost_alert_threshold_usd
                        and self.total_cost_usd > self.cost_alert_threshold_usd):
                    logger.warning(
                        "成本超过阈值 $%.2f, 当前累计 $%.2f",
                        self.cost_alert_threshold_usd, self.total_cost_usd,
                    )

            if self.verbose:
                logger.info(
                    "Claude API call %d: in=%d, out=%d, cost=$%.4f, total=$%.4f",
                    self.total_calls, in_tok, out_tok,
                    (in_tok * pricing["input"] + out_tok * pricing["output"]) / 1_000_000 if pricing else 0,
                    self.total_cost_usd,
                )

        return response_text

    def _enforce_rate_limit(self) -> None:
        """Sliding window rate limit."""
        now = time.time()
        cutoff = now - 60.0
        self._recent_call_times = [t for t in self._recent_call_times if t > cutoff]
        if len(self._recent_call_times) >= self.rate_limit_calls_per_minute:
            # 等到最早调用过 60s
            wait = self._recent_call_times[0] + 60.0 - now
            if wait > 0:
                if self.verbose:
                    logger.info(
                        "Rate limit: 等待 %.1fs (当前窗口 %d/%d calls)",
                        wait, len(self._recent_call_times),
                        self.rate_limit_calls_per_minute,
                    )
                time.sleep(wait)

    def get_stats(self) -> dict[str, Any]:
        """返回当前累计统计."""
        return {
            "model": self.model,
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd,
            "total_retries": self.total_retries,
            "pricing": MODEL_PRICING.get(self.model),
        }

    def reset_stats(self) -> None:
        """重置统计 (新会话用)."""
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.total_retries = 0
        self._recent_call_times = []
