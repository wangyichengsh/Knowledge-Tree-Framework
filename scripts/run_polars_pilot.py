#!/usr/bin/env python3
"""
scripts/run_polars_pilot.py
=============================

Phase 4.2 Stage 1 - Polars 10 题 × 3 cond pilot 实验 runner.

设计 (基于 Phase 4.1 经验 + Stage 0 实测):
  - 仅 30 generations, 不需要 budget guard / 复杂 resume
  - 3 conditions: A_null / B_hybrid / F_irrelevant (Phase 4.1 已证 C/D/E 与 B 无差异)
  - 严格 eval (polars_benchmark.eval_strict): API name + 真实代码执行
  - 与 Phase 4.1 R1 一致: 4-bit 量化 + LoRA (用户类型三校准要求)

输出:
  jsonl: 每行 1 generation, 含 prompt / response / extracted_code / eval / token_count
  
后续:
  pilot 数据看 baseline + RAG 效应趋势, 决定是否扩展到 50 题

用法:
  # R1 + LoRA (4-bit, 与 Phase 4.1 一致)
  python scripts/run_polars_pilot.py \\
    --base-model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \\
    --explorer-lora models/explorer-grpo-sanity/checkpoint-50 \\
    --use-int4 \\
    --tree-json knowledge_tree/docs/polars/tree.json \\
    --output polars_pilot_r1.jsonl \\
    --api-key $ANTHROPIC_API_KEY

  # Nemotron (4-bit, Stage 0 实测 work)
  python scripts/run_polars_pilot.py \\
    --base-model ./models/nemotron-nano-9b-v2 \\
    --use-int4 \\
    --tree-json knowledge_tree/docs/polars/tree.json \\
    --output polars_pilot_nemo.jsonl \\
    --api-key $ANTHROPIC_API_KEY

PROTO 关联:
  PROTO-7.1 (grep 复用): 复用 retrievers / load_model / generate / inject 逻辑
  PROTO-7.4 (实测校准): 严格 eval + 真实 exec verify
  PROTO-7.20 (每轮 grep outputs): 在跑前确认 polars_benchmark + tree.json 都已就绪
  PROTO-7.22 (显存监控): nvidia-smi 真实读数
"""

import argparse
import gc
import json
import logging
import os
import re
import subprocess
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polars_benchmark import (
    POLARS_BENCHMARK_TASKS,
    PolarsBenchmarkTask,
    eval_strict,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Prompt 构造 (Polars 适配, Phase 4.1 inject 风格)
# ============================================================================

def build_polars_prompt(task: PolarsBenchmarkTask,
                         retrieved_nodes: list = None) -> tuple[str, int]:
    """
    构造 Polars 任务 prompt.

    Args:
        task: PolarsBenchmarkTask
        retrieved_nodes: None=baseline; 否则是 retrieved KnowledgeNode list

    Returns:
        (prompt_text, inject_chars)
    """
    if not retrieved_nodes:
        # A_null: baseline (无 inject)
        prompt = (
            f"{task.task_description}\n\n"
            f"Wrap your final code in ```python ... ```."
        )
        return prompt, 0

    # RAG 注入 (复用 Phase 4.1 inject 格式, 仅替换 'mathematical' → 'Polars')
    inject_parts = []
    total_chars = 0
    for node in retrieved_nodes:
        text = node.llm_inject_text()
        inject_parts.append(text)
        total_chars += len(text)

    inject_block = "\n\n---\n\n".join(inject_parts)

    prompt = (
        f"Here are relevant Polars 1.0+ API references that may help:\n\n"
        f"{inject_block}\n\n"
        f"---\n\n"
        f"Now complete this task:\n\n"
        f"{task.task_description}\n\n"
        f"Wrap your final code in ```python ... ```."
    )
    return prompt, total_chars


# ============================================================================
# 代码提取
# ============================================================================

def extract_code_block(response: str) -> str:
    """
    从 ```python ... ``` 中提取代码.
    
    修复 (基于 R1 pilot bug 诊断):
      1. 取最后一个 python block (R1 reasoning model 可能在 thinking 阶段写 code)
      2. 清理 Qwen tokenizer 的 Ġ artefact (Ġ = byte-level BPE space prefix)
      3. 清理 Ċ artefact (newline)
    """
    # 先清理 tokenizer artefacts (Qwen/GPT-2 style BPE)
    clean = response.replace('Ġ', ' ').replace('Ċ', '\n')
    
    # 取最后一个 python code block (R1 reasoning 可能多个)
    patterns = [
        (r"```python\s*\n(.*?)```", True),  # python first
        (r"```py\s*\n(.*?)```", True),
        (r"```\s*\n(.*?)```", True),
    ]
    for pat, _ in patterns:
        matches = re.findall(pat, clean, re.DOTALL)
        if matches:
            # 取最后一个 (final answer)
            return matches[-1].strip()
    return clean.strip()


# ============================================================================
# 显存监控
# ============================================================================

def get_nvidia_smi_vram() -> Optional[float]:
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
# 模型加载 (复用 polars_sanity_check 逻辑)
# ============================================================================

def load_model(base_model: str, explorer_lora: Optional[str], use_int4: bool):
    """加载模型 + LoRA. 返回 (model, tokenizer)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    vram_before = get_nvidia_smi_vram()
    print(f"  加载前 vram: {vram_before:.1f} GB" if vram_before else "")

    load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if use_int4:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        del load_kwargs["torch_dtype"]

    # tokenizer (Nemotron 需 trust_remote_code)
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    try:
        model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, trust_remote_code=True, **load_kwargs,
        )

    if explorer_lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, explorer_lora)

    vram_after = get_nvidia_smi_vram()
    if vram_after is not None:
        print(f"  加载后 vram: {vram_after:.1f} GB")

    return model, tokenizer


def generate_one(model, tokenizer, prompt: str,
                  max_new_tokens: int = 4096) -> tuple[str, int, str]:
    """generate. 返回 (response_text, response_token_len, status)."""
    import torch

    messages = [{"role": "user", "content": prompt}]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
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
    response_len = response_ids.shape[0]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

    status = "inside_truncated" if response_len >= max_new_tokens else "complete"

    del outputs, inputs, response_ids
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return response_text, response_len, status


# ============================================================================
# Pilot main
# ============================================================================

def run_pilot(args):
    """跑 10 题 × 3 cond = 30 generations."""
    from knowledge_tree.storage import JSONStorage
    from knowledge_tree.core import KnowledgeTree
    from knowledge_tree.retrievers import (
        NullRetriever, HybridRetriever, IrrelevantRetriever,
    )
    from knowledge_tree.llm_clients import ClaudeCallable

    # === 加载 tree ===
    print(f"\n[1/4] 加载 Polars KTF tree: {args.tree_json}")
    storage = JSONStorage(args.tree_json, create_if_missing=False)
    tree = KnowledgeTree(storage.list_all())
    print(f"  {len(tree)} 节点加载")

    # === 加载 retrievers ===
    print(f"\n[2/4] 初始化 retrievers")
    claude = ClaudeCallable(
        api_key=args.api_key,
        model="claude-sonnet-4-6",
        max_tokens=2048,
        temperature=0.3,
        verbose=False,
    )

    # Polars 适配的 rerank prompt (替换默认 math prompt)
    POLARS_RERANK_PROMPT = """You are re-ranking candidate Polars 1.0+ API concepts for a code generation task. Given the candidates below, select the {top_k} that are MOST directly applicable.

## Task
{query}

## Candidates
{candidates_listing}

## Instructions
Pick the {top_k} candidates that:
1. Have APIs / methods that apply DIRECTLY to this task (not just topically related)
2. Are at the right specificity level (prefer specific API documentation over general concepts when applicable)
3. Specifically address Polars 1.0+ syntax (not pandas, not Polars 0.x)

Respond ONLY with a JSON object:
{{"selected_ids": ["id1", "id2", "id3"]}}"""

    retrievers = {
        "A_null": NullRetriever(tree),
        "B_hybrid": HybridRetriever(
            tree, llm_callable=claude,
            bm25_top_n=8, tree_top_n=5, rerank_input_size=8,
            rerank_prompt_template=POLARS_RERANK_PROMPT,
        ),
        "F_irrelevant": IrrelevantRetriever(tree, seed=args.seed),
    }
    print(f"  retrievers ready: {list(retrievers.keys())}")
    print(f"  ⚠️ Note: HybridRetriever internal TreeNavigationRetriever uses default")
    print(f"    'math problem' nav prompt (Phase 4.2 known limit; BM25+LLM rerank兜底)")

    # === 加载模型 ===
    print(f"\n[3/4] 加载模型: {args.base_model}")
    print(f"  use_int4: {args.use_int4}, LoRA: {args.explorer_lora}")
    model, tokenizer = load_model(args.base_model, args.explorer_lora, args.use_int4)
    print(f"  ✓ model loaded")

    # === 跑实验 ===
    print(f"\n[4/4] 跑 pilot 实验 ({len(POLARS_BENCHMARK_TASKS)} 题 × {len(retrievers)} cond)")
    print()

    records = []
    start = time.time()
    total_gens = len(POLARS_BENCHMARK_TASKS) * len(retrievers)
    gen_idx = 0
    cond_order = ["A_null", "B_hybrid", "F_irrelevant"]

    for task in POLARS_BENCHMARK_TASKS:
        for cond_name in cond_order:
            gen_idx += 1
            retriever = retrievers[cond_name]
            t_start = time.time()

            # 检索 (A_null 返回空)
            t_ret = time.time()
            try:
                if cond_name == "A_null":
                    retrieved = []
                else:
                    # 用 task description 作为 query
                    retrieved = retriever.retrieve(task.task_description, top_k=3)
            except Exception as e:
                logger.warning("Retrieval failed for %s/%s: %s",
                                task.name, cond_name, e)
                retrieved = []
            ret_time = time.time() - t_ret

            # 构造 prompt
            prompt, inject_chars = build_polars_prompt(task, retrieved)

            # generate
            t_gen = time.time()
            try:
                response, response_len, status = generate_one(
                    model, tokenizer, prompt, max_new_tokens=args.max_new_tokens,
                )
            except Exception as e:
                logger.error("Generation failed for %s/%s: %s", task.name, cond_name, e)
                response, response_len, status = "", 0, f"error: {e}"
            gen_time = time.time() - t_gen

            # 提取代码 + eval
            code = extract_code_block(response)
            eval_result = eval_strict(code, task, run_code=True)

            vram = get_nvidia_smi_vram()
            vram_str = f"  vram={vram:.1f}GB" if vram else ""

            # 输出进度
            marker = "✓" if eval_result['is_correct'] else "✗"
            elapsed = (time.time() - start) / 60
            eta = (total_gens - gen_idx) * (elapsed / gen_idx)
            print(
                f"  [{gen_idx}/{total_gens}] {marker} task={task.task_id}({task.name[:18]:<18}) "
                f"cond={cond_name:<13} hits={len(eval_result['expected_hits'])}/"
                f"{len(eval_result['expected_hits']) + len(eval_result['expected_misses'])} "
                f"anti={len(eval_result['anti_hits'])} "
                f"runt={'✓' if eval_result['runtime_passed'] else '✗'} "
                f"t={gen_time:.0f}s ret={ret_time:.1f}s len={response_len} "
                f"ETA={eta:.0f}min{vram_str}"
            )
            if eval_result['expected_misses']:
                print(f"      missing: {eval_result['expected_misses']}")
            if eval_result['anti_hits']:
                print(f"      anti: {eval_result['anti_hits']}")
            if eval_result['runtime_error']:
                print(f"      runtime err: {eval_result['runtime_error'][:80]}")
            if eval_result['test_failure']:
                print(f"      test fail: {eval_result['test_failure'][:80]}")

            record = {
                "task_id": task.task_id,
                "task_name": task.name,
                "category": task.category,
                "difficulty": task.difficulty,
                "condition": cond_name,
                "retrieved_node_ids": [n.id for n in retrieved],
                "inject_chars": inject_chars,
                "prompt_chars": len(prompt),
                "response": response,
                "extracted_code": code,
                "response_token_len": response_len,
                "status": status,
                "eval": eval_result,
                "is_correct": eval_result['is_correct'],
                "ret_time_s": round(ret_time, 1),
                "gen_time_s": round(gen_time, 1),
                "model": args.base_model,
                "use_int4": args.use_int4,
                "lora": args.explorer_lora,
            }
            records.append(record)

            # 立即写 jsonl (防中断丢失)
            with open(args.output, 'a') as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # === 总结 ===
    elapsed_min = (time.time() - start) / 60
    print(f"\n{'='*70}")
    print(f"Pilot 完成: {len(records)} generations in {elapsed_min:.1f} min")
    print(f"{'='*70}")

    # Accuracy by condition
    from collections import defaultdict
    by_cond = defaultdict(lambda: {'correct': 0, 'total': 0})
    for r in records:
        by_cond[r['condition']]['total'] += 1
        if r['is_correct']:
            by_cond[r['condition']]['correct'] += 1

    print("\nAccuracy by condition:")
    for cond in cond_order:
        s = by_cond[cond]
        print(f"  {cond:<15} {s['correct']}/{s['total']} = "
              f"{s['correct']/s['total']*100:.0f}%")

    # Save
    print(f"\n  Output: {args.output}")
    if hasattr(claude, 'get_stats'):
        stats = claude.get_stats()
        print(f"  Claude API: {stats.get('total_calls')} calls, "
              f"${stats.get('total_cost_usd', 0):.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", required=True,
                        help="模型路径 (HF repo 或本地)")
    parser.add_argument("--explorer-lora", default=None,
                        help="LoRA checkpoint (R1 用, Nemo 不需要)")
    parser.add_argument("--use-int4", action="store_true",
                        help="4-bit 量化 (与 Phase 4.1 一致, R1 + Nemo 都用)")
    parser.add_argument("--tree-json", required=True,
                        help="Polars KTF tree JSON")
    parser.add_argument("--max-new-tokens", type=int, default=4096,
                        help="代码任务一般够")
    parser.add_argument("--seed", type=int, default=42,
                        help="IrrelevantRetriever seed")
    parser.add_argument("--api-key", default=None,
                        help="Anthropic API key (HybridRetriever 用)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 清空旧 jsonl (pilot 不需要 resume)
    if os.path.exists(args.output):
        os.remove(args.output)
        print(f"清空旧 {args.output}")

    print("=" * 70)
    print("Phase 4.2 Stage 1 Pilot - Polars 10 题 × 3 conditions")
    print("=" * 70)

    run_pilot(args)


if __name__ == "__main__":
    main()
