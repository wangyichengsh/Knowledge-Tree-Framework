#!/usr/bin/env python3
"""
scripts/polars_sanity_check.py
================================

Phase 4.2 Stage 0 - Sanity Check (1 天工作量, 不投入大工程之前的验证)

目的:
  在投入 50 题 mini benchmark 之前, 先用 5 题快速验证 H-M 框架前提:
    (i) R1-Distill 在 Polars 1.0+ 任务上是否真的"不会"
    (ii) Nemotron-Nano 在 5090 32GB 上能否加载 + 跑通
  
  这是 PROTO-7.4 + PROTO-7.6 的应用: 先实测前提, 再投入大工程.

设计 (基于 H-M 框架):
  - Polars 1.0 发布: 2024.07
  - R1-Distill base Qwen2.5 cutoff: 2023.10
  - 满足: R1 完全没见过 Polars 1.0 (前提 (iii) 模型确实缺概念)
  
  - Nemotron-Nano-9B-v2 paper: 2025.08, 训练数据 ~2025 中
  - Polars 1.0 (2024.07) < Nemo cutoff → Nemo 可能见过
  - 创造 H-M 跨模型对照场景

5 题选择原则:
  - 覆盖 Polars 核心 API (lazy, expression, join, group, streaming)
  - 每题有明确测试用例 (可 automatic eval)
  - 难度梯度: 简单 → 复杂

判断准则:
  R1-Distill baseline:
    0/5 或 1/5 → 确认 H-M 前提, 进 Stage 1 ✅
    2-3/5     → 部分知识, 选更 specific 的 unseen API
    4-5/5     → R1 实际见过 Polars (推翻假设, 换库)

  Nemotron-Nano:
    跑通且 vram OK → Stage 2 可行 ✅
    OOM → 用 4-bit 量化 (bnb) 或换更小模型

用法:
  # R1 baseline (5 题, ~30-50 分钟)
  python scripts/polars_sanity_check.py \\
    --base-model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \\
    --explorer-lora models/explorer-grpo-sanity/checkpoint-50 \\
    --output polars_sanity_r1.jsonl
  
  # Nemotron 加载 + 跑 5 题 (~30 分钟, INT4 量化)
  python scripts/polars_sanity_check.py \\
    --base-model nvidia/NVIDIA-Nemotron-Nano-9B-v2 \\
    --use-int4 \\
    --output polars_sanity_nemo.jsonl
  
  # Dry-run (mock model, 测 plumbing)
  python scripts/polars_sanity_check.py --dry-run --output /tmp/dry.jsonl

PROTO 关联:
  PROTO-7.4 (实测校准): 5 题快速验证, 不上来就大工程
  PROTO-7.6 (不基于"应该 work"): R1 cutoff 假设必须实测验证
  PROTO-7.7 (错题分诊优先): 5 题区分模型能力边界
  PROTO-7.22 (显存监控用 nvidia-smi): Nemotron 加载时实测
"""

import argparse
import gc
import json
import logging
import os
import subprocess
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


# ============================================================================
# 5 题 Polars sanity benchmark
# ============================================================================
# 设计原则:
#   - 每题给一个具体任务描述
#   - 提供输入数据格式 (作为 prompt 一部分)
#   - 给出 expected 输出 (用于评判)
#   - 关键 API 必须 Polars 1.0+ 才稳定支持
#
# 难度梯度:
#   1: 基础 lazy frame collect
#   2: expressions + alias
#   3: group_by_dynamic (1.0+ 重命名, 之前是 groupby_dynamic)
#   4: join with validate (1.0+ 新参数)
#   5: streaming sink_parquet (1.0+ 高级特性)

POLARS_SANITY_TASKS = [
    {
        "id": 1,
        "name": "lazy_collect",
        "difficulty": "easy",
        "task": (
            "Using Polars 1.0+, write Python code that:\n"
            "1. Reads a CSV file '/tmp/people.csv' lazily (without loading into memory)\n"
            "2. Filters rows where the 'age' column > 30\n"
            "3. Collects the result into a DataFrame\n"
            "Return only the code, no explanation. Use modern Polars API."
        ),
        "expected_apis": ["pl.scan_csv", ".filter", "pl.col", ".collect"],
        "anti_patterns": ["pd.read_csv", "pandas"],  # 不应使用 pandas
        "ground_truth": (
            "import polars as pl\n"
            "result = (\n"
            "    pl.scan_csv('/tmp/people.csv')\n"
            "    .filter(pl.col('age') > 30)\n"
            "    .collect()\n"
            ")\n"
        ),
    },
    {
        "id": 2,
        "name": "expressions_alias",
        "difficulty": "easy",
        "task": (
            "Using Polars 1.0+, write Python code to compute the mean of column 'price' "
            "from DataFrame `df`, and rename the result column to 'avg_price'. "
            "Use Polars expression API.\n"
            "Return only the code, no explanation."
        ),
        "expected_apis": ["pl.col", ".mean()", ".alias"],
        "anti_patterns": ["df['price'].mean()"],  # pandas style
        "ground_truth": (
            "import polars as pl\n"
            "result = df.select(pl.col('price').mean().alias('avg_price'))\n"
        ),
    },
    {
        "id": 3,
        "name": "group_by_dynamic",
        "difficulty": "medium",
        "task": (
            "Using Polars 1.0+ (NOT 0.x), perform time-based grouping on DataFrame `df`:\n"
            "- Group by column 'timestamp' with daily windows ('1d')\n"
            "- Aggregate column 'value' with sum\n"
            "Return only the code, no explanation. Use modern Polars 1.0 API "
            "(method name changed from earlier versions)."
        ),
        # 关键 API: group_by_dynamic (1.0+), 不是 groupby_dynamic (0.x)
        "expected_apis": ["group_by_dynamic", "every"],
        "anti_patterns": ["groupby_dynamic", "df.resample"],
        "ground_truth": (
            "import polars as pl\n"
            "result = df.group_by_dynamic('timestamp', every='1d').agg(pl.col('value').sum())\n"
        ),
    },
    {
        "id": 4,
        "name": "join_validate",
        "difficulty": "medium",
        "task": (
            "Using Polars 1.0+, perform a left join of `df_a` and `df_b` on column 'id'. "
            "Use the `validate` parameter to ensure 1-to-1 mapping "
            "(this parameter was added in Polars 1.0).\n"
            "Return only the code, no explanation."
        ),
        # validate='1:1' 是 Polars 1.0+ 加入的
        "expected_apis": [".join", "how='left'", "validate"],
        "anti_patterns": ["pd.merge", "df_a.merge"],
        "ground_truth": (
            "import polars as pl\n"
            "result = df_a.join(df_b, on='id', how='left', validate='1:1')\n"
        ),
    },
    {
        "id": 5,
        "name": "streaming_sink",
        "difficulty": "hard",
        "task": (
            "Using Polars 1.0+ streaming engine, write `df` (a LazyFrame) to a parquet file "
            "'/tmp/output.parquet' in streaming mode (without loading all into memory). "
            "Include full statistics in the parquet metadata.\n"
            "Return only the code, no explanation."
        ),
        # sink_parquet with statistics='full' 是 Polars 1.0+
        "expected_apis": ["sink_parquet", "statistics"],
        "anti_patterns": ["write_parquet", "df.to_parquet"],
        "ground_truth": (
            "import polars as pl\n"
            "df.sink_parquet('/tmp/output.parquet', statistics='full')\n"
        ),
    },
]


# ============================================================================
# Prompt 构造
# ============================================================================

def build_polars_prompt(task: dict) -> str:
    """构造 chat prompt (与 aime_evaluator_dryrun 风格一致)."""
    return (
        f"{task['task']}\n\n"
        f"Wrap your final answer in a ```python code block.```"
    )


# ============================================================================
# Eval 逻辑
# ============================================================================

def extract_code_block(response: str) -> str:
    """提取 ```python ... ``` 中的代码.
    
    修复 (基于 R1 pilot bug 诊断):
      1. 取最后一个 python block (R1 reasoning model 可能在 thinking 阶段写 code)
      2. 清理 Qwen tokenizer 的 Ġ artefact (Ġ = byte-level BPE space prefix)
      3. 清理 Ċ artefact (newline)
    """
    import re
    # 先清理 tokenizer artefacts (Qwen/GPT-2 style BPE)
    clean = response.replace('Ġ', ' ').replace('Ċ', '\n')
    
    patterns = [
        r"```python\s*\n(.*?)```",
        r"```py\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]
    for pat in patterns:
        matches = re.findall(pat, clean, re.DOTALL)
        if matches:
            # 取最后一个 (final answer for reasoning models)
            return matches[-1].strip()
    return clean.strip()


def eval_polars_task(code: str, task: dict) -> dict:
    """
    评估一个 Polars 任务的代码.

    Returns:
        {
            'has_expected_apis': bool,  # 含所有 expected API
            'has_anti_patterns': bool,  # 含禁止 API (pandas-style)
            'is_correct': bool,         # 综合判断 (expected 且不含 anti-pattern)
            'expected_hits': list[str],  # 命中哪些 API
            'expected_misses': list[str], # 漏掉哪些 API
            'anti_hits': list[str],      # 命中哪些 anti-pattern
        }
    """
    code_lower = code.lower()
    
    # 检查 expected APIs (大小写敏感, 因为 Python API 名敏感)
    expected_hits = []
    expected_misses = []
    for api in task['expected_apis']:
        if api.lower() in code_lower:
            expected_hits.append(api)
        else:
            expected_misses.append(api)
    
    # 检查 anti-patterns
    anti_hits = []
    for ap in task['anti_patterns']:
        if ap.lower() in code_lower:
            anti_hits.append(ap)
    
    has_expected_apis = len(expected_misses) == 0
    has_anti_patterns = len(anti_hits) > 0
    
    # 综合判断:
    #   - 必须含所有 expected_apis
    #   - 不应含 anti_patterns
    # 这是 lenient eval, 实际语法/运行正确性需要执行测试
    is_correct = has_expected_apis and not has_anti_patterns
    
    return {
        'has_expected_apis': has_expected_apis,
        'has_anti_patterns': has_anti_patterns,
        'is_correct': is_correct,
        'expected_hits': expected_hits,
        'expected_misses': expected_misses,
        'anti_hits': anti_hits,
    }


# ============================================================================
# 显存监控 (PROTO-7.22)
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
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return None


# ============================================================================
# 模型加载 + 跑题
# ============================================================================

def run_sanity_check(
    base_model: str,
    explorer_lora: Optional[str],
    use_int4: bool,
    max_new_tokens: int,
    output_path: str,
    dry_run: bool,
):
    """跑 5 题 sanity check."""

    # === 加载模型 ===
    if dry_run:
        print("\n[Stage 0] Dry-run: 用 mock model")
        model, tokenizer = None, None
    else:
        print(f"\n[Stage 0] 加载模型: {base_model}")
        print(f"  use_int4: {use_int4}")
        print(f"  explorer_lora: {explorer_lora}")

        # 监控加载前后显存
        vram_before = get_nvidia_smi_vram()
        if vram_before is not None:
            print(f"  加载前 vram: {vram_before:.1f} GB")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
        }
        if use_int4:
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                )
                # 量化时不要 dtype
                del load_kwargs["torch_dtype"]
            except ImportError:
                logger.warning("bitsandbytes 不可用, 跳过量化")

        # 处理 trust_remote_code (Nemotron 可能需要)
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model)
        except Exception as e:
            if "trust_remote_code" in str(e).lower() or "custom" in str(e).lower():
                logger.info("加载 tokenizer 时需要 trust_remote_code=True, 重试")
                tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            else:
                raise

        try:
            model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
        except Exception as e:
            if "trust_remote_code" in str(e).lower() or "custom" in str(e).lower():
                logger.info("加载 model 时需要 trust_remote_code=True, 重试")
                model = AutoModelForCausalLM.from_pretrained(
                    base_model, trust_remote_code=True, **load_kwargs,
                )
            else:
                raise

        # 加 LoRA
        if explorer_lora:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, explorer_lora)

        vram_after = get_nvidia_smi_vram()
        if vram_after is not None:
            delta = (vram_after - (vram_before or 0))
            print(f"  加载后 vram: {vram_after:.1f} GB (Δ +{delta:.1f} GB)")
            if vram_after > 30:
                print(f"  ⚠️ 加载后 vram > 30GB, 接近 5090 32GB 上限")

    # === 跑 5 题 ===
    print(f"\n[Stage 1] 跑 {len(POLARS_SANITY_TASKS)} 题 sanity check\n")
    records = []
    start_time = time.time()

    for i, task in enumerate(POLARS_SANITY_TASKS):
        prompt = build_polars_prompt(task)
        t_start = time.time()

        if dry_run:
            # mock 返回
            response = (
                "```python\n"
                "import polars as pl\n"
                "result = pl.scan_csv('/tmp/people.csv').filter(pl.col('age') > 30).collect()\n"
                "```"
            )
            response_length = 50
            status = "complete"
        else:
            # 真模型 generate
            response, response_length, status = _generate(
                model, tokenizer, prompt, max_new_tokens,
            )

        t_elapsed = time.time() - t_start

        # 提取代码 + eval
        code = extract_code_block(response)
        eval_result = eval_polars_task(code, task)

        # 监控 vram
        vram = get_nvidia_smi_vram()
        vram_str = f"  vram={vram:.1f}GB" if vram else ""

        record = {
            "task_id": task['id'],
            "task_name": task['name'],
            "difficulty": task['difficulty'],
            "prompt": prompt,
            "response": response,
            "extracted_code": code,
            "response_length": response_length,
            "time_s": round(t_elapsed, 1),
            "status": status,
            "eval": eval_result,
            "is_correct": eval_result['is_correct'],
            "model": base_model,
        }
        records.append(record)

        marker = "✓" if eval_result['is_correct'] else "✗"
        print(
            f"  [{i+1}/{len(POLARS_SANITY_TASKS)}] {marker} {task['name']} "
            f"({task['difficulty']}): "
            f"hits={len(eval_result['expected_hits'])}/{len(task['expected_apis'])} "
            f"anti={len(eval_result['anti_hits'])} "
            f"t={t_elapsed:.0f}s len={response_length}{vram_str}"
        )
        if eval_result['expected_misses']:
            print(f"      Missing APIs: {eval_result['expected_misses']}")
        if eval_result['anti_hits']:
            print(f"      ⚠️ Anti-patterns: {eval_result['anti_hits']}")

    # === 写出 + 总结 ===
    with open(output_path, 'w') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'='*70}")
    print(f"Sanity Check 完成 in {(time.time()-start_time)/60:.1f} min")
    print(f"{'='*70}")
    print(f"\nResults: {output_path}")
    print()

    n_correct = sum(1 for r in records if r['is_correct'])
    print(f"Accuracy: {n_correct}/{len(records)} = {n_correct/len(records)*100:.0f}%")
    print()

    # === 判断准则 ===
    print("Decision (基于 H-M 框架):")
    if n_correct <= 1:
        print(f"  ✅ {n_correct}/5 → 确认 H-M 前提 (模型不会做 Polars 1.0)")
        print(f"     → 进 Stage 1: build Polars 50 题 mini benchmark")
    elif n_correct >= 4:
        print(f"  ❌ {n_correct}/5 → 模型实际见过 Polars (推翻 H-M 前提)")
        print(f"     → 换 candidate library (Pydantic V3 / FastAPI 0.130+ 等)")
    else:
        print(f"  🟡 {n_correct}/5 → 部分知识, 需要选更 specific 的 unseen API")
        print(f"     → 看哪些题对, 哪些题错, 调整任务设计")

    # === 分诊 ===
    print()
    print("分诊详情:")
    for r in records:
        e = r['eval']
        if not r['is_correct']:
            reasons = []
            if e['expected_misses']:
                reasons.append(f"missing {len(e['expected_misses'])} APIs")
            if e['anti_hits']:
                reasons.append(f"用了 anti-pattern {e['anti_hits']}")
            print(f"  ✗ {r['task_name']}: {' / '.join(reasons)}")

    # === 显存最终状态 ===
    if not dry_run:
        vram = get_nvidia_smi_vram()
        if vram:
            print(f"\n最终 vram (PROTO-7.22 nvidia-smi): {vram:.1f} GB")


def _generate(model, tokenizer, prompt: str, max_new_tokens: int) -> tuple[str, int, str]:
    """generate one - 简化版 (不依赖 phase2_mcts)."""
    import torch

    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        # Fallback: 直接用 prompt
        formatted = prompt

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response_ids = outputs[0, input_len:]
    response_length = response_ids.shape[0]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

    # 状态: 是否撞顶
    if response_length >= max_new_tokens:
        status = "inside_truncated"
    else:
        status = "complete"

    # 清理 (PROTO-7.3)
    del outputs, inputs, response_ids
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return response_text, response_length, status


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-model", default=None,
                        help="模型路径. 例如 deepseek-ai/DeepSeek-R1-Distill-Qwen-14B "
                             "或 nvidia/NVIDIA-Nemotron-Nano-9B-v2")
    parser.add_argument("--explorer-lora", default=None,
                        help="LoRA checkpoint (可选, 仅 DeepSeek 用)")
    parser.add_argument("--use-int4", action="store_true",
                        help="4-bit 量化 (bnb), 推荐 Nemotron-Nano 用")
    parser.add_argument("--max-new-tokens", type=int, default=4096,
                        help="代码任务一般 1-2K tokens, 4096 留 headroom")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 70)
    print("Phase 4.2 Stage 0 - Polars Sanity Check")
    print("=" * 70)

    if not args.dry_run and not args.base_model:
        print("ERROR: --base-model required (or --dry-run)")
        sys.exit(1)

    run_sanity_check(
        base_model=args.base_model,
        explorer_lora=args.explorer_lora,
        use_int4=args.use_int4,
        max_new_tokens=args.max_new_tokens,
        output_path=args.output,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
