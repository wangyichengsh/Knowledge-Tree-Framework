#!/usr/bin/env python3
"""
scripts/run_polars_local_retriever.py
========================================

Phase 4.2 Stage 1.3a 实验: 用 Nemotron 替代 Claude API 做 retriever.

架构 (顺序加载, 5090 32GB 关键):
  ┌────────────────────────────────────────────────────┐
  │ Stage A: Retrieval (Nemo retriever)                 │
  │   1. Load Nemotron (~8GB INT4)                     │
  │   2. For each task: retrieve via HybridRetriever   │
  │   3. Write retrieved_node_ids to cache file        │
  │   4. Unload Nemotron (释放显存)                    │
  └────────────────────────────────────────────────────┘
                          ↓
  ┌────────────────────────────────────────────────────┐
  │ Stage B: Generation (R1 + LoRA generator)           │
  │   1. Load R1 (~10GB INT4 + LoRA)                   │
  │   2. For each task:                                │
  │      - Read retrieved nodes from cache             │
  │      - Build prompt with inject                    │
  │      - Generate code                               │
  │      - Eval                                        │
  └────────────────────────────────────────────────────┘

主要对比:
  vs run_polars_pilot.py (Claude API rerank): 看 B-F 效应是否保持
  
H-M 框架预测:
  - B_hybrid (Nemo retriever) ≈ B_hybrid (Claude retriever)
  - 节省 Claude API ~$0.50/run
  - Stage A 时间 ~5-10 min (Nemo 14x R1 速度), Stage B 一致

用法:
  # 标准跑法
  python scripts/run_polars_local_retriever.py \\
    --retriever-model ./models/nemotron-nano-9b-v2 \\
    --generator-model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \\
    --generator-lora models/explorer-grpo-sanity/checkpoint-50 \\
    --tree-json knowledge_tree/docs/polars/tree.json \\
    --output polars_50_local_retriever.jsonl
  
  # 仅跑 Stage A (retrieve, 可中断后继续)
  python scripts/run_polars_local_retriever.py --stage-a-only \\
    --retriever-model ./models/nemotron-nano-9b-v2 \\
    --tree-json knowledge_tree/docs/polars/tree.json \\
    --retrieval-cache /tmp/polars_retrieval.jsonl

  # 用现有 cache 跑 Stage B
  python scripts/run_polars_local_retriever.py --stage-b-only \\
    --generator-model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \\
    --generator-lora models/explorer-grpo-sanity/checkpoint-50 \\
    --retrieval-cache /tmp/polars_retrieval.jsonl \\
    --output polars_50_local_retriever.jsonl

PROTO 关联:
  PROTO-7.1 (grep 复用): 复用 polars_benchmark / run_polars_pilot 大部分逻辑
  PROTO-7.4 (实测校准): 与 Claude rerank baseline paired CI 对照
  PROTO-7.20 (每轮 grep outputs): retrieval_cache 分阶段写, 便于检查
  PROTO-7.22 (显存监控): load/unload 之间打印 vram
"""

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polars_benchmark import POLARS_BENCHMARK_TASKS, eval_strict
from run_polars_pilot import (
    build_polars_prompt,
    extract_code_block,
    get_nvidia_smi_vram,
)

logger = logging.getLogger(__name__)


# Polars 适配的 rerank prompt (与 run_polars_pilot 一致)
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


# ============================================================================
# Stage A: Retrieval (用 Nemo 替代 Claude)
# ============================================================================

def run_stage_a_retrieval(
    retriever_model: str,
    tree_json: str,
    output_cache: str,
    seed: int,
    verbose: bool,
) -> dict:
    """跑 Stage A: 用本地 Nemo 做 retrieval, 写 cache."""
    from knowledge_tree.storage import JSONStorage
    from knowledge_tree.core import KnowledgeTree
    from knowledge_tree.retrievers import (
        NullRetriever, HybridRetriever, IrrelevantRetriever,
    )
    from knowledge_tree.local_model_clients import LocalModelCallable

    print("\n" + "=" * 70)
    print("Stage A: Retrieval (用 Nemotron 替代 Claude API)")
    print("=" * 70)

    # Load tree
    print(f"\n[A1] 加载 KTF tree: {tree_json}")
    storage = JSONStorage(tree_json, create_if_missing=False)
    tree = KnowledgeTree(storage.list_all())
    print(f"  {len(tree)} 节点加载")

    # Load Nemo retriever
    print(f"\n[A2] 加载 Nemotron retriever: {retriever_model}")
    nemo = LocalModelCallable(
        base_model=retriever_model,
        use_int4=True,
        max_new_tokens=512,  # retriever 输出短
        temperature=0.3,  # 低温, 确定性
        keep_thinking=False,
        verbose=verbose,
        lazy_load=False,
    )
    print(f"  ✓ Nemo loaded, vram={nemo.vram_at_load:.1f} GB")

    # Init retrievers (B_hybrid 用 Nemo, A_null + F_irrelevant 不用 LLM)
    retrievers = {
        "A_null": NullRetriever(tree),
        "B_hybrid": HybridRetriever(
            tree, llm_callable=nemo,
            bm25_top_n=8, tree_top_n=5, rerank_input_size=8,
            rerank_prompt_template=POLARS_RERANK_PROMPT,
        ),
        "F_irrelevant": IrrelevantRetriever(tree, seed=seed),
    }

    # 跑 retrieval (50 题 × 3 cond = 150 retrieve)
    print(f"\n[A3] 跑 retrieval ({len(POLARS_BENCHMARK_TASKS)} 题 × {len(retrievers)} cond)")

    cache_records = []
    start = time.time()
    total = len(POLARS_BENCHMARK_TASKS) * len(retrievers)
    idx = 0

    # 清空旧 cache
    if os.path.exists(output_cache):
        os.remove(output_cache)

    for task in POLARS_BENCHMARK_TASKS:
        for cond_name, retriever in retrievers.items():
            idx += 1
            t_start = time.time()

            try:
                if cond_name == "A_null":
                    retrieved = []
                else:
                    retrieved = retriever.retrieve(task.task_description, top_k=3)
            except Exception as e:
                logger.warning("Retrieval failed %s/%s: %s",
                                task.name, cond_name, e)
                retrieved = []

            ret_time = time.time() - t_start

            record = {
                "task_id": task.task_id,
                "task_name": task.name,
                "condition": cond_name,
                "retrieved_node_ids": [n.id for n in retrieved],
                "retrieval_time_s": round(ret_time, 2),
                "retriever_model": retriever_model,
            }
            cache_records.append(record)

            # 立即写 (防中断)
            with open(output_cache, 'a') as f:
                f.write(json.dumps(record) + "\n")

            elapsed = (time.time() - start) / 60
            eta = (total - idx) * (elapsed / idx) if idx > 0 else 0
            vram = get_nvidia_smi_vram()
            print(
                f"  [{idx}/{total}] task={task.task_id} cond={cond_name:<13} "
                f"retrieved={len(retrieved)} t={ret_time:.1f}s "
                f"vram={vram:.1f}GB ETA={eta:.1f}min" if vram else
                f"  [{idx}/{total}] task={task.task_id} cond={cond_name:<13} "
                f"retrieved={len(retrieved)} t={ret_time:.1f}s ETA={eta:.1f}min"
            )

    elapsed_min = (time.time() - start) / 60
    print(f"\n[A4] Stage A 完成 in {elapsed_min:.1f} min")
    nemo_stats = nemo.get_stats()
    print(f"  Nemo calls: {nemo_stats['total_calls']}")
    print(f"  Nemo cost: $0 (本地)")
    print(f"  Retrieval cache: {output_cache}")

    # Unload Nemo (Stage 1.3 关键: 释放显存给 R1)
    print(f"\n[A5] Unload Nemo (释放显存给 Stage B)")
    nemo.unload()
    vram_after = get_nvidia_smi_vram()
    print(f"  vram after unload: {vram_after:.1f} GB")

    return {
        "stage_a_min": elapsed_min,
        "nemo_calls": nemo_stats['total_calls'],
        "cache_path": output_cache,
        "n_records": len(cache_records),
    }


# ============================================================================
# Stage B: Generation (R1 + LoRA, 与 run_polars_pilot 一致)
# ============================================================================

def run_stage_b_generation(
    generator_model: str,
    generator_lora: str,
    tree_json: str,
    retrieval_cache: str,
    output: str,
    max_new_tokens: int,
    verbose: bool,
) -> dict:
    """跑 Stage B: 用 R1+LoRA 生成 code, 读 cache 中的 retrieved nodes."""
    from knowledge_tree.storage import JSONStorage
    from knowledge_tree.core import KnowledgeTree
    from knowledge_tree.local_model_clients import LocalModelCallable

    print("\n" + "=" * 70)
    print("Stage B: Generation (R1 + LoRA)")
    print("=" * 70)

    # Load tree (从 cache 中获取 node 内容)
    print(f"\n[B1] 加载 KTF tree: {tree_json}")
    storage = JSONStorage(tree_json, create_if_missing=False)
    tree = KnowledgeTree(storage.list_all())
    print(f"  {len(tree)} 节点加载")

    # Load retrieval cache
    print(f"\n[B2] 读 retrieval cache: {retrieval_cache}")
    if not os.path.exists(retrieval_cache):
        raise FileNotFoundError(f"Cache 不存在: {retrieval_cache}")
    cache_records = []
    with open(retrieval_cache) as f:
        for line in f:
            line = line.strip()
            if line:
                cache_records.append(json.loads(line))
    print(f"  {len(cache_records)} retrieval records 加载")

    # 建 task lookup
    tasks_by_id = {t.task_id: t for t in POLARS_BENCHMARK_TASKS}

    # Load R1 generator
    print(f"\n[B3] 加载 R1 generator: {generator_model}")
    print(f"  LoRA: {generator_lora}")
    r1 = LocalModelCallable(
        base_model=generator_model,
        explorer_lora=generator_lora,
        use_int4=True,
        max_new_tokens=max_new_tokens,
        temperature=0.6,  # 与 polars_sanity_check / run_polars_pilot 一致
        keep_thinking=True,  # 保留 thinking, extract_code_block 自己处理
        verbose=verbose,
        lazy_load=False,
    )
    print(f"  ✓ R1 loaded, vram={r1.vram_at_load:.1f} GB")

    # 跑 generation
    print(f"\n[B4] 跑 generation ({len(cache_records)} records)")

    # 清空旧 output
    if os.path.exists(output):
        os.remove(output)

    start = time.time()
    total = len(cache_records)

    for idx, cache_record in enumerate(cache_records, 1):
        task = tasks_by_id[cache_record['task_id']]
        condition = cache_record['condition']

        # 用 cache 中的 retrieved IDs 获取 nodes
        retrieved_nodes = []
        for nid in cache_record['retrieved_node_ids']:
            if tree.has_node(nid):
                retrieved_nodes.append(tree.get_node(nid))

        # 构造 prompt (复用 run_polars_pilot.build_polars_prompt)
        prompt, inject_chars = build_polars_prompt(task, retrieved_nodes)

        # Generate
        t_gen = time.time()
        try:
            response = r1(prompt)
            response_len = len(r1.tokenizer.encode(response))
            status = "complete"
        except Exception as e:
            logger.error("Gen failed: %s", e)
            response = ""
            response_len = 0
            status = f"error: {e}"
        gen_time = time.time() - t_gen

        # Eval
        code = extract_code_block(response)
        eval_result = eval_strict(code, task, run_code=True)

        elapsed_min = (time.time() - start) / 60
        eta = (total - idx) * (elapsed_min / idx)
        vram = get_nvidia_smi_vram()
        marker = "✓" if eval_result['is_correct'] else "✗"
        print(
            f"  [{idx}/{total}] {marker} task={task.task_id} cond={condition:<13} "
            f"hits={len(eval_result['expected_hits'])}/{len(eval_result['expected_hits']) + len(eval_result['expected_misses'])} "
            f"runt={'✓' if eval_result['runtime_passed'] else '✗'} "
            f"t={gen_time:.0f}s ETA={eta:.0f}min vram={vram:.1f}GB"
        )

        # 写 output
        record = {
            "task_id": task.task_id,
            "task_name": task.name,
            "category": task.category,
            "difficulty": task.difficulty,
            "condition": condition,
            "retrieved_node_ids": cache_record['retrieved_node_ids'],
            "inject_chars": inject_chars,
            "prompt_chars": len(prompt),
            "response": response,
            "extracted_code": code,
            "response_token_len": response_len,
            "status": status,
            "eval": eval_result,
            "is_correct": eval_result['is_correct'],
            "ret_time_s": cache_record.get('retrieval_time_s', 0),
            "gen_time_s": round(gen_time, 1),
            "generator_model": generator_model,
            "generator_lora": generator_lora,
            "retriever_model": cache_record.get('retriever_model', 'unknown'),
        }
        with open(output, 'a') as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    elapsed_min = (time.time() - start) / 60

    print(f"\n[B5] Stage B 完成 in {elapsed_min:.1f} min")
    print(f"  R1 calls: {r1.get_stats()['total_calls']}")
    print(f"  Output: {output}")

    # 统计
    from collections import defaultdict
    by_cond = defaultdict(lambda: {'c': 0, 't': 0})
    with open(output) as f:
        for line in f:
            r = json.loads(line)
            by_cond[r['condition']]['t'] += 1
            if r['is_correct']:
                by_cond[r['condition']]['c'] += 1

    print(f"\nAccuracy by condition:")
    for cond in ['A_null', 'B_hybrid', 'F_irrelevant']:
        s = by_cond[cond]
        print(f"  {cond:<15} {s['c']}/{s['t']} = {s['c']/s['t']*100 if s['t'] else 0:.0f}%")

    r1.unload()

    return {
        "stage_b_min": elapsed_min,
        "r1_calls": r1.get_stats()['total_calls'],
        "output_path": output,
    }


# ============================================================================
# Main: 顺序跑 Stage A + Stage B
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Models
    parser.add_argument("--retriever-model", default="./models/nemotron-nano-9b-v2",
                        help="Retriever 模型 (本地路径或 HF repo)")
    parser.add_argument("--generator-model",
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    parser.add_argument("--generator-lora",
                        default="models/explorer-grpo-sanity/checkpoint-50")

    # Data
    parser.add_argument("--tree-json", required=True, help="Polars KTF tree JSON")
    parser.add_argument("--retrieval-cache", default="/tmp/polars_retrieval_cache.jsonl",
                        help="Stage A 输出 / Stage B 输入")
    parser.add_argument("--output", default="polars_50_local_retriever.jsonl")

    # Generation params (与 run_polars_pilot 一致)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)

    # Stage control
    parser.add_argument("--stage-a-only", action="store_true",
                        help="仅跑 Stage A retrieve, 写 cache")
    parser.add_argument("--stage-b-only", action="store_true",
                        help="仅跑 Stage B generate, 读 cache")

    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 70)
    print("Phase 4.2 Stage 1.3a: Local Retriever (顺序加载架构)")
    print("=" * 70)
    print(f"Retriever: {args.retriever_model}")
    print(f"Generator: {args.generator_model} + LoRA {args.generator_lora}")
    print(f"Tree: {args.tree_json}")
    print(f"Retrieval cache: {args.retrieval_cache}")
    print(f"Output: {args.output}")

    total_start = time.time()

    # === Stage A ===
    if not args.stage_b_only:
        stage_a_stats = run_stage_a_retrieval(
            retriever_model=args.retriever_model,
            tree_json=args.tree_json,
            output_cache=args.retrieval_cache,
            seed=args.seed,
            verbose=args.verbose,
        )
        if args.stage_a_only:
            print(f"\n[Stage A 完成, --stage-a-only 退出]")
            return

    # === Stage B ===
    stage_b_stats = run_stage_b_generation(
        generator_model=args.generator_model,
        generator_lora=args.generator_lora,
        tree_json=args.tree_json,
        retrieval_cache=args.retrieval_cache,
        output=args.output,
        max_new_tokens=args.max_new_tokens,
        verbose=args.verbose,
    )

    total_min = (time.time() - total_start) / 60
    print(f"\n{'=' * 70}")
    print(f"全部完成 in {total_min:.1f} min")
    print(f"{'=' * 70}")
    print(f"  Output: {args.output}")
    print(f"  vs run_polars_pilot.py 节省 Claude API ~$0.5")


if __name__ == "__main__":
    main()
