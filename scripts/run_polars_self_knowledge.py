#!/usr/bin/env python3
"""
scripts/run_polars_self_knowledge.py
======================================

Phase 4.2 Stage 1.3c: Self-Knowledge Filter 实验.

用户指令: "Stage 1.3c 时 R1 + Nemo 两种 self-judge 都跑, 对比"

设计 (基于 STAGE_1_3C_SELF_KNOWLEDGE_DESIGN.md 候选 A):
  Condition G1 (R1 self-judge):
    - HybridRetriever (Nemo rerank) → top-5
    - R1 self-knowledge filter → keep "NO" → top-3
    - R1 generate code
  
  Condition G2 (Nemo self-judge):
    - HybridRetriever (Nemo rerank) → top-5
    - Nemo self-knowledge filter → keep "NO" → top-3
    - R1 generate code

实验对照 (paired CI):
  vs B_hybrid baseline (Phase 4.2: R1 92%)

H-M 框架预测:
  - G1 (R1 self-judge): 准确度可能高 (主模型 + LoRA-trained 有较好自知?)
    但 R1 慢 (~13s/judge, 50题×5节点×2cond=500 judge ≈ 110 min Stage A)
  - G2 (Nemo self-judge): 速度快 (~1s/judge), 但自知准确度可能不如 R1
  - 都可能不如 B_hybrid (因为 LLM zero-shot 自知不准 — SR-RAG/SKILL-RAG 论文已警告)

架构: 顺序加载 (复用 Stage 1.3a)
  Stage A: Retriever (Nemo) + Self-Judge (R1 or Nemo)
    → 写 selected_node_ids cache
  Stage B: R1 generator → 读 cache → 跑

用法:
  # Step 1: G1 (R1 self-judge)
  python scripts/run_polars_self_knowledge.py \\
    --judge-model r1 \\
    --tree-json knowledge_tree/docs/polars/tree.json \\
    --output polars_50_g1_r1_judge.jsonl

  # Step 2: G2 (Nemo self-judge)
  python scripts/run_polars_self_knowledge.py \\
    --judge-model nemo \\
    --tree-json knowledge_tree/docs/polars/tree.json \\
    --output polars_50_g2_nemo_judge.jsonl

PROTO 关联:
  PROTO-7.1 (grep 复用): 复用 LocalModelCallable, HybridRetriever, SelfKnowledgeFilter
  PROTO-7.4 (实测校准): paired CI vs B_hybrid baseline
  PROTO-7.20 (grep outputs): retrieval_cache 分阶段写, 可中断重启
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


# Polars 适配 rerank prompt (与 Stage 1.3a 一致)
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


def run_stage_a_self_judge(
    judge_model_choice: str,  # 'r1' or 'nemo'
    retriever_model: str,
    generator_model: str,
    generator_lora: str,
    tree_json: str,
    output_cache: str,
    top_k_initial: int,
    top_k_final: int,
    min_keep: int,
    verbose: bool,
) -> dict:
    """
    Stage A: Retrieval + Self-Knowledge Filter.
    
    模型加载策略:
      - 'r1' as judge: 加载 R1 (做 judge), 卸载, Stage B 再加载 R1 (generation)
                       问题: R1 加载慢, 这里加载 2 次
                       优化: judge 用 R1, 然后保持 R1 加载, Stage B 直接复用
      - 'nemo' as judge: 加载 Nemo (做 retrieve+judge), 卸载, Stage B 加载 R1
                          架构与 Stage 1.3a 完全一致
    """
    from knowledge_tree.storage import JSONStorage
    from knowledge_tree.core import KnowledgeTree
    from knowledge_tree.retrievers import NullRetriever, IrrelevantRetriever
    from knowledge_tree.self_knowledge_filter import (
        HybridWithSelfKnowledgeRetriever, SelfKnowledgeFilter,
    )
    from knowledge_tree.retrievers import HybridRetriever
    from knowledge_tree.local_model_clients import LocalModelCallable

    print("\n" + "=" * 70)
    print(f"Stage A: Self-Knowledge Filter (judge_model={judge_model_choice})")
    print("=" * 70)

    # Load tree
    print(f"\n[A1] 加载 KTF tree: {tree_json}")
    storage = JSONStorage(tree_json, create_if_missing=False)
    tree = KnowledgeTree(storage.list_all())
    print(f"  {len(tree)} 节点加载")

    # Load Nemo (rerank, 总是用)
    print(f"\n[A2] 加载 Nemotron retriever (做 hybrid rerank)")
    nemo = LocalModelCallable(
        base_model=retriever_model,
        use_int4=True, max_new_tokens=512, temperature=0.3,
        keep_thinking=False, verbose=verbose, lazy_load=False,
    )
    print(f"  ✓ Nemo loaded, vram={nemo.vram_at_load:.1f} GB")

    # 决定 judge LLM
    if judge_model_choice == 'r1':
        print(f"\n[A2b] 加载 R1 (做 self-judge)")
        r1_judge = LocalModelCallable(
            base_model=generator_model,
            explorer_lora=generator_lora,
            use_int4=True, max_new_tokens=64,  # judge 只需短 YES/NO
            temperature=0.3,  # 低温, 确定性
            keep_thinking=False, verbose=verbose, lazy_load=False,
        )
        print(f"  ✓ R1 loaded, vram={r1_judge.vram_at_load:.1f} GB")
        # Stage B 会复用这个 R1 (不卸载)
        judge_llm = r1_judge
        keep_judge_loaded = True
    elif judge_model_choice == 'nemo':
        # Nemo 同时做 rerank + judge
        judge_llm = nemo
        keep_judge_loaded = False
    else:
        raise ValueError(f"Unknown judge_model: {judge_model_choice}")

    # 构建 retrievers
    null_ret = NullRetriever(tree)
    irrelevant_ret = IrrelevantRetriever(tree, seed=42)
    hybrid_sk_ret = HybridWithSelfKnowledgeRetriever(
        tree,
        rerank_llm=nemo,           # 总是 Nemo 做 rerank
        filter_llm=judge_llm,      # R1 or Nemo 做 self-judge
        top_k_initial=top_k_initial,
        min_keep=min_keep,
        rerank_prompt_template=POLARS_RERANK_PROMPT,
        verbose=verbose,
    )
    # 也跑普通 hybrid 做对照
    hybrid_ret = HybridRetriever(
        tree, llm_callable=nemo,
        bm25_top_n=8, tree_top_n=5, rerank_input_size=8,
        rerank_prompt_template=POLARS_RERANK_PROMPT,
    )

    retrievers = {
        "A_null": null_ret,
        "B_hybrid": hybrid_ret,            # baseline
        "G_self_knowledge": hybrid_sk_ret,  # 新 condition
        "F_irrelevant": irrelevant_ret,
    }

    # 清空旧 cache
    if os.path.exists(output_cache):
        os.remove(output_cache)

    print(f"\n[A3] 跑 retrieval + self-judge ({len(POLARS_BENCHMARK_TASKS)} 题 × {len(retrievers)} cond)")
    start = time.time()
    total = len(POLARS_BENCHMARK_TASKS) * len(retrievers)
    idx = 0

    for task in POLARS_BENCHMARK_TASKS:
        for cond_name, retriever in retrievers.items():
            idx += 1
            t_s = time.time()
            try:
                if cond_name == "A_null":
                    retrieved = []
                else:
                    retrieved = retriever.retrieve(task.task_description, top_k=top_k_final)
            except Exception as e:
                logger.warning("Retrieve failed %s/%s: %s", task.name, cond_name, e)
                retrieved = []
            
            t_e = time.time() - t_s
            record = {
                "task_id": task.task_id,
                "task_name": task.name,
                "condition": cond_name,
                "retrieved_node_ids": [n.id for n in retrieved],
                "retrieval_time_s": round(t_e, 2),
                "judge_model": judge_model_choice if cond_name == "G_self_knowledge" else None,
            }
            # 写 cache
            with open(output_cache, 'a') as f:
                f.write(json.dumps(record) + "\n")

            elapsed = (time.time() - start) / 60
            eta = (total - idx) * (elapsed / idx) if idx > 0 else 0
            vram = get_nvidia_smi_vram() or -1
            print(
                f"  [{idx}/{total}] task={task.task_id} cond={cond_name:<17} "
                f"got={len(retrieved)} t={t_e:.1f}s vram={vram:.1f}GB ETA={eta:.1f}min"
            )

    elapsed_min = (time.time() - start) / 60
    print(f"\n[A4] Stage A 完成 in {elapsed_min:.1f} min")
    print(f"  Nemo calls: {nemo.get_stats()['total_calls']}")
    if judge_model_choice == 'r1':
        print(f"  R1 (judge) calls: {r1_judge.get_stats()['total_calls']}")
    # Stage 1.3c specific: filter stats
    sk_stats = hybrid_sk_ret.get_stats()
    print(f"  Self-Knowledge filter: queried={sk_stats['total_queried']}, "
          f"kept={sk_stats['total_kept']}, filtered_out={sk_stats['total_filtered_out']} "
          f"(rate={sk_stats['filter_rate']:.2%})")

    # 卸载策略
    if judge_model_choice == 'nemo':
        print(f"\n[A5] Unload Nemo (释放给 Stage B R1)")
        nemo.unload()
        return {"keep_loaded": None, "elapsed_min": elapsed_min}
    else:
        # R1 已加载, 直接传给 Stage B
        print(f"\n[A5] R1 保持加载 (Stage B 直接复用)")
        nemo.unload()  # Nemo 卸载
        return {"keep_loaded": r1_judge, "elapsed_min": elapsed_min}


def run_stage_b_generation(
    generator_model: str,
    generator_lora: str,
    tree_json: str,
    retrieval_cache: str,
    output: str,
    max_new_tokens: int,
    preloaded_r1=None,
    verbose: bool=False,
) -> dict:
    """Stage B: R1 generation (复用 Stage 1.3a 大部分)."""
    from knowledge_tree.storage import JSONStorage
    from knowledge_tree.core import KnowledgeTree
    from knowledge_tree.local_model_clients import LocalModelCallable

    print("\n" + "=" * 70)
    print("Stage B: Generation (R1 + LoRA)")
    print("=" * 70)

    print(f"\n[B1] 加载 tree")
    storage = JSONStorage(tree_json, create_if_missing=False)
    tree = KnowledgeTree(storage.list_all())

    print(f"\n[B2] 读 retrieval cache: {retrieval_cache}")
    cache_records = [json.loads(l) for l in open(retrieval_cache) if l.strip()]
    print(f"  {len(cache_records)} records")

    tasks_by_id = {t.task_id: t for t in POLARS_BENCHMARK_TASKS}

    if preloaded_r1 is not None:
        # 复用 Stage A 加载的 R1 (但 reconfigure 为 generation mode)
        print(f"\n[B3] 复用 Stage A 已加载 R1, 切换为 generation 模式")
        # 改 max_new_tokens (judge 用 64, generation 需要 4096)
        preloaded_r1.max_new_tokens = max_new_tokens
        preloaded_r1.temperature = 0.6  # generation 温度
        preloaded_r1.keep_thinking = True  # generation 需要 thinking, extract_code 处理
        r1 = preloaded_r1
        r1.reset_stats()
    else:
        print(f"\n[B3] 加载 R1 generator")
        r1 = LocalModelCallable(
            base_model=generator_model,
            explorer_lora=generator_lora,
            use_int4=True, max_new_tokens=max_new_tokens,
            temperature=0.6, keep_thinking=True,
            verbose=verbose, lazy_load=False,
        )
    
    print(f"  ✓ R1 ready, vram={get_nvidia_smi_vram():.1f} GB")

    # 清空 output
    if os.path.exists(output):
        os.remove(output)

    print(f"\n[B4] 跑 generation ({len(cache_records)} records)")
    start = time.time()
    total = len(cache_records)

    for idx, cr in enumerate(cache_records, 1):
        task = tasks_by_id[cr['task_id']]
        condition = cr['condition']

        retrieved_nodes = []
        for nid in cr['retrieved_node_ids']:
            if tree.has_node(nid):
                retrieved_nodes.append(tree.get_node(nid))

        prompt, inject_chars = build_polars_prompt(task, retrieved_nodes)

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

        code = extract_code_block(response)
        eval_result = eval_strict(code, task, run_code=True)

        elapsed_min = (time.time() - start) / 60
        eta = (total - idx) * (elapsed_min / idx)
        vram = get_nvidia_smi_vram() or -1
        marker = "✓" if eval_result['is_correct'] else "✗"
        print(
            f"  [{idx}/{total}] {marker} task={task.task_id} cond={condition:<17} "
            f"hits={len(eval_result['expected_hits'])}/{len(eval_result['expected_hits']) + len(eval_result['expected_misses'])} "
            f"t={gen_time:.0f}s ETA={eta:.0f}min vram={vram:.1f}GB"
        )

        record = {
            "task_id": task.task_id, "task_name": task.name,
            "category": task.category, "difficulty": task.difficulty,
            "condition": condition,
            "retrieved_node_ids": cr['retrieved_node_ids'],
            "inject_chars": inject_chars,
            "prompt_chars": len(prompt),
            "response": response,
            "extracted_code": code,
            "response_token_len": response_len,
            "status": status,
            "eval": eval_result,
            "is_correct": eval_result['is_correct'],
            "ret_time_s": cr.get('retrieval_time_s', 0),
            "gen_time_s": round(gen_time, 1),
            "judge_model": cr.get('judge_model'),
        }
        with open(output, 'a') as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    elapsed_min = (time.time() - start) / 60
    print(f"\n[B5] Stage B 完成 in {elapsed_min:.1f} min")

    # Accuracy
    from collections import defaultdict
    by_c = defaultdict(lambda: {'c': 0, 't': 0})
    for line in open(output):
        r = json.loads(line)
        by_c[r['condition']]['t'] += 1
        if r['is_correct']:
            by_c[r['condition']]['c'] += 1

    print(f"\nAccuracy by condition:")
    for cond in ['A_null', 'B_hybrid', 'G_self_knowledge', 'F_irrelevant']:
        s = by_c[cond]
        if s['t'] > 0:
            print(f"  {cond:<20} {s['c']}/{s['t']} = {s['c']/s['t']*100:.0f}%")

    r1.unload()
    return {"elapsed_min": elapsed_min}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge-model", choices=['r1', 'nemo'], required=True,
                        help="Self-judge LLM: 'r1' (主模型自判) 或 'nemo' (Nemo 自判)")
    parser.add_argument("--retriever-model", default="./models/nemotron-nano-9b-v2")
    parser.add_argument("--generator-model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    parser.add_argument("--generator-lora", default="models/explorer-grpo-sanity/checkpoint-50")
    parser.add_argument("--tree-json", required=True)
    parser.add_argument("--retrieval-cache", default=None,
                        help="default: /tmp/polars_sk_cache_<judge>.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k-initial", type=int, default=5,
                        help="Hybrid retrieve 前 K (默认 5, filter 后 → 3)")
    parser.add_argument("--top-k-final", type=int, default=3)
    parser.add_argument("--min-keep", type=int, default=1,
                        help="Filter 后最少保留几个 (默认 1)")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--stage-a-only", action="store_true")
    parser.add_argument("--stage-b-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.retrieval_cache is None:
        args.retrieval_cache = f"/tmp/polars_sk_cache_{args.judge_model}.jsonl"

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 70)
    print(f"Stage 1.3c: Self-Knowledge Filter (judge={args.judge_model})")
    print("=" * 70)
    print(f"  Retriever:  {args.retriever_model}")
    print(f"  Generator:  {args.generator_model} + LoRA {args.generator_lora}")
    print(f"  Judge:      {args.judge_model}")
    print(f"  Tree:       {args.tree_json}")
    print(f"  Cache:      {args.retrieval_cache}")
    print(f"  Output:     {args.output}")
    print(f"  top_k_initial={args.top_k_initial}, top_k_final={args.top_k_final}, min_keep={args.min_keep}")

    total_start = time.time()
    preloaded_r1 = None

    if not args.stage_b_only:
        stage_a_result = run_stage_a_self_judge(
            judge_model_choice=args.judge_model,
            retriever_model=args.retriever_model,
            generator_model=args.generator_model,
            generator_lora=args.generator_lora,
            tree_json=args.tree_json,
            output_cache=args.retrieval_cache,
            top_k_initial=args.top_k_initial,
            top_k_final=args.top_k_final,
            min_keep=args.min_keep,
            verbose=args.verbose,
        )
        preloaded_r1 = stage_a_result.get("keep_loaded")
        if args.stage_a_only:
            print(f"\n[Stage A 完成, --stage-a-only 退出]")
            return

    stage_b_result = run_stage_b_generation(
        generator_model=args.generator_model,
        generator_lora=args.generator_lora,
        tree_json=args.tree_json,
        retrieval_cache=args.retrieval_cache,
        output=args.output,
        max_new_tokens=args.max_new_tokens,
        preloaded_r1=preloaded_r1,
        verbose=args.verbose,
    )

    total_min = (time.time() - total_start) / 60
    print(f"\n{'=' * 70}")
    print(f"全部完成 in {total_min:.1f} min")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
