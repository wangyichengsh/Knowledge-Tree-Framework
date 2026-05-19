"""
knowledge_tree/local_model_clients.py
=======================================

LocalModelCallable: 本地模型 (HuggingFace transformers) 的 LLMCallable 实现.

设计 (Phase 4.2 Stage 1.3a):
  - 接口与 ClaudeCallable 完全兼容 (Callable[[str], str])
  - 复用 run_polars_pilot 已实测的 load_model / generate_one 逻辑
  - 支持: R1-Distill 4-bit + LoRA, Nemotron 4-bit
  - 支持 lazy load (避免立即加载占显存)
  - 支持 unload (释放显存, Stage A retrieve → Stage B generate 关键)
  - tokenizer artefact 处理 (Ġ/Ċ + </think>, 复用 phase2_mcts 方案)
  - 统计 (calls, tokens) 与 ClaudeCallable 一致

关键架构: 顺序加载
  ┌─────────────────────────────────────────────────────────┐
  │ Stage A: Retrieval (Nemo retriever 加载)                 │
  │   - retriever_llm = LocalModelCallable("Nemotron")      │
  │   - retriever_llm.load()  # 占 ~8GB                     │
  │   - for task: retrieve(...)                             │
  │   - retriever_llm.unload()  # 释放显存                  │
  └─────────────────────────────────────────────────────────┘
                              ↓
  ┌─────────────────────────────────────────────────────────┐
  │ Stage B: Generation (R1+LoRA generator 加载)             │
  │   - generator_llm = LocalModelCallable("R1", lora=...)  │
  │   - generator_llm.load()  # 占 ~10GB                    │
  │   - for task: generate(...)                             │
  └─────────────────────────────────────────────────────────┘

PROTO 关联:
  PROTO-7.1 (grep 复用): 复用 run_polars_pilot load_model / generate_one
  PROTO-7.4 (实测校准): 与 ClaudeCallable 接口一致, 可直接替换
  PROTO-7.16 (借理念不依赖工具): LLMCallable 抽象隔离 SDK 细节
  PROTO-7.22 (nvidia-smi 显存监控): load/unload 实测 vram
"""

import gc
import logging
import re
import subprocess
import time
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Tokenizer artefact cleanup (复用 phase2_mcts 方案, 与 run_polars_pilot 一致)
# ============================================================================

def clean_response(response: str, keep_thinking: bool = False) -> str:
    """
    清理 LLM response 中的 tokenizer artefact + thinking content.

    复用 phase2_mcts.py 方案 (PROTO-7.1 grep 复用):
      1. Ġ → space (Qwen/GPT-2 byte-level BPE space prefix)
      2. Ċ → newline (byte-level BPE newline)
      3. <think>...</think> 切分: 取 </think> 之后内容
         (R1-Distill / Nemotron 等 reasoning model 实际格式)

    Args:
        response: raw response text from model.generate
        keep_thinking: 若 True 不切分 </think>, 保留 reasoning

    Returns:
        cleaned response text
    """
    # Step 1: 清理 Qwen byte-level BPE artefacts
    text = response.replace('Ġ', ' ').replace('Ċ', '\n')

    if keep_thinking:
        return text

    # Step 2: </think> 切分
    if '</think>' in text:
        # 仅保留 </think> 之后 (final answer)
        text = text.split('</think>', 1)[1]
    else:
        # 兜底: 用配对正则过滤 <think>...</think>
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    return text.strip()


# ============================================================================
# 显存监控 (复用 run_polars_pilot, 与 PROTO-7.22 一致)
# ============================================================================

def get_nvidia_smi_vram() -> Optional[float]:
    """用 nvidia-smi 读真实进程显存 (GB). 失败返回 None."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            mb = int(result.stdout.strip().split("\n")[0])
            return mb / 1024
    except Exception:
        pass
    return None


# ============================================================================
# LocalModelCallable
# ============================================================================

class LocalModelCallable:
    """
    本地模型 LLM 调用接口, 实现 LLMCallable 协议.

    示例:
        >>> # 用法 1: 主动 load (推荐, 显存可见)
        >>> nemo = LocalModelCallable("./models/nemotron-nano-9b-v2", use_int4=True)
        >>> nemo.load()
        >>> response = nemo("What is Polars scan_csv?")
        >>> nemo.unload()  # 释放显存

        >>> # 用法 2: lazy load (首次 call 时加载)
        >>> r1 = LocalModelCallable(
        ...     "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        ...     explorer_lora="models/explorer-grpo-sanity/checkpoint-50",
        ...     use_int4=True,
        ... )
        >>> response = r1("question")  # 自动加载

    实例属性 (统计, 与 ClaudeCallable 接口一致):
        total_calls: 累计调用次数
        total_input_tokens: 累计 input tokens
        total_output_tokens: 累计 output tokens
        total_cost_usd: 0.0 (本地无 cost, 与 ClaudeCallable 接口对齐)
        total_retries: 累计 retry 次数 (本地通常 0, 偶尔 OOM 算 retry)
        is_loaded: 模型是否已加载
    """

    def __init__(
        self,
        base_model: str,
        explorer_lora: Optional[str] = None,
        use_int4: bool = True,
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
        top_p: float = 0.95,
        keep_thinking: bool = False,
        lazy_load: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            base_model: 模型路径 (HF repo ID 或本地路径)
            explorer_lora: LoRA checkpoint 路径 (可选, R1 用)
            use_int4: 是否 4-bit 量化 (Phase 4.1 一致默认 True)
            max_new_tokens: 单次 generation 最大输出 token
                            注意: 与 ClaudeCallable max_tokens 对应
                            retriever 用途建议 256-1024 (输出短)
                            generator 用途建议 4096+ (输出长 code)
            temperature: 采样温度 (默认 0.3, 适合 retriever 任务)
            top_p: nucleus sampling (默认 0.95)
            keep_thinking: 是否保留 <think>...</think> 内容
                          retriever 用途: False (要 final answer)
                          debug 用途: True (看推理过程)
            lazy_load: 是否首次 call 时才加载 (默认 False, 主动加载)
            verbose: 详细日志
        """
        self.base_model = base_model
        self.explorer_lora = explorer_lora
        self.use_int4 = use_int4
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.keep_thinking = keep_thinking
        self.verbose = verbose

        # 模型对象 (lazy load)
        self.model = None
        self.tokenizer = None
        self.is_loaded = False

        # 统计 (与 ClaudeCallable 接口一致)
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0  # 本地无 cost
        self.total_retries = 0

        # vram 监控
        self.vram_at_load: Optional[float] = None

        if not lazy_load:
            self.load()

    # ========================================================================
    # 加载 / 卸载 (Stage 1.3 顺序加载架构关键)
    # ========================================================================

    def load(self) -> None:
        """加载模型 + LoRA. 复用 run_polars_pilot 的 load_model 逻辑."""
        if self.is_loaded:
            logger.info("LocalModelCallable: 模型已加载, 跳过")
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "transformers 未安装. pip install transformers torch"
            ) from e

        vram_before = get_nvidia_smi_vram()
        if self.verbose:
            logger.info(
                "LocalModelCallable: 加载 %s (use_int4=%s)",
                self.base_model, self.use_int4,
            )
            if vram_before is not None:
                logger.info("  加载前 vram: %.1f GB", vram_before)

        load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
        if self.use_int4:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            del load_kwargs["torch_dtype"]

        # Tokenizer (Nemotron 需 trust_remote_code)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model, trust_remote_code=True,
            )

        # Model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model, **load_kwargs,
            )
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model, trust_remote_code=True, **load_kwargs,
            )

        # LoRA (可选)
        if self.explorer_lora:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.explorer_lora)

        self.is_loaded = True

        vram_after = get_nvidia_smi_vram()
        if vram_after is not None:
            self.vram_at_load = vram_after
            if self.verbose:
                delta = vram_after - (vram_before or 0)
                logger.info(
                    "  加载后 vram: %.1f GB (Δ +%.1f GB)", vram_after, delta,
                )

    def unload(self) -> None:
        """
        卸载模型, 释放显存.

        关键: Stage A retrieve 后, 卸载 retriever 模型, 再加载 generator 模型.
              这是 Stage 1.3 顺序加载架构的核心.
        """
        if not self.is_loaded:
            return

        if self.verbose:
            vram_before = get_nvidia_smi_vram()
            logger.info(
                "LocalModelCallable: 卸载 %s (当前 vram %.1f GB)",
                self.base_model, vram_before or -1,
            )

        # 删除引用
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        self.is_loaded = False

        # 强制 GC + CUDA cache clear
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

        if self.verbose:
            vram_after = get_nvidia_smi_vram()
            logger.info("  卸载后 vram: %.1f GB", vram_after or -1)

    def __del__(self):
        """析构时自动卸载."""
        try:
            self.unload()
        except Exception:
            pass

    # ========================================================================
    # LLMCallable 协议: __call__(prompt) -> response
    # ========================================================================

    def __call__(self, prompt: str) -> str:
        """
        生成 response. 实现 LLMCallable 协议.

        Args:
            prompt: input prompt (chat template 自动应用)

        Returns:
            cleaned response text (Ġ/Ċ 处理 + </think> 切分)

        统计 (副作用):
            self.total_calls += 1
            self.total_input_tokens += input length
            self.total_output_tokens += response length
        """
        if not prompt:
            raise ValueError("prompt 不能为空")

        if not self.is_loaded:
            self.load()

        import torch

        # Apply chat template
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            formatted = prompt  # fallback for older tokenizers

        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        input_len = inputs.input_ids.shape[1]

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=(self.temperature > 0),
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        except torch.cuda.OutOfMemoryError as e:
            # OOM: 清理 + 重试一次 (max_new_tokens 减半)
            self.total_retries += 1
            logger.warning(
                "LocalModelCallable: CUDA OOM, retry with half max_new_tokens (%d → %d)",
                self.max_new_tokens, self.max_new_tokens // 2,
            )
            gc.collect()
            torch.cuda.empty_cache()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens // 2,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=(self.temperature > 0),
                    pad_token_id=self.tokenizer.eos_token_id,
                )

        response_ids = outputs[0, input_len:]
        response_len = response_ids.shape[0]
        response_text = self.tokenizer.decode(
            response_ids, skip_special_tokens=True,
        )

        # 统计
        self.total_calls += 1
        self.total_input_tokens += input_len
        self.total_output_tokens += response_len

        # 清理 (PROTO-7.3)
        del outputs, inputs, response_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 清理 tokenizer artefact + </think> 切分
        cleaned = clean_response(response_text, keep_thinking=self.keep_thinking)
        return cleaned

    # ========================================================================
    # 统计接口 (与 ClaudeCallable 一致)
    # ========================================================================

    def get_stats(self) -> dict:
        """返回统计 dict (接口与 ClaudeCallable.get_stats 一致)."""
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd,  # 0 (本地)
            "total_retries": self.total_retries,
            "is_loaded": self.is_loaded,
            "vram_at_load_gb": self.vram_at_load,
            "model": self.base_model,
            "lora": self.explorer_lora,
        }

    def reset_stats(self) -> None:
        """重置统计 (用于多阶段实验)."""
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_retries = 0


# ============================================================================
# 便利工厂函数
# ============================================================================

def make_nemotron_retriever(
    model_path: str = "./models/nemotron-nano-9b-v2",
    max_new_tokens: int = 512,
    verbose: bool = False,
) -> LocalModelCallable:
    """
    便利工厂: Nemotron-Nano-9B-v2 用作 retriever (Stage 1.3a 推荐).

    参数:
        max_new_tokens=512 (retriever 任务输出短, 不需要 4096)
        temperature=0.3 (低温, 确定性 retrieval)

    显存: ~8 GB (INT4)
    速度: ~14x R1 (Stage 0 实测)
    """
    return LocalModelCallable(
        base_model=model_path,
        use_int4=True,
        max_new_tokens=max_new_tokens,
        temperature=0.3,
        keep_thinking=False,  # retriever 要 final answer
        verbose=verbose,
    )


def make_r1_generator(
    model_path: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    lora_path: Optional[str] = "models/explorer-grpo-sanity/checkpoint-50",
    max_new_tokens: int = 4096,
    verbose: bool = False,
) -> LocalModelCallable:
    """
    便利工厂: R1-Distill + LoRA 用作 generator (Phase 4.1 / 4.2 一致).

    参数:
        max_new_tokens=4096 (代码生成需要长输出)
        temperature=0.6 (与 polars_sanity_check 一致)

    显存: ~10 GB (INT4 + LoRA)
    """
    return LocalModelCallable(
        base_model=model_path,
        explorer_lora=lora_path,
        use_int4=True,
        max_new_tokens=max_new_tokens,
        temperature=0.6,
        keep_thinking=False,
        verbose=verbose,
    )
