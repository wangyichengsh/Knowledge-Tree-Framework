#!/usr/bin/env python3
"""
scripts/day5_step6_swebench_eval.py
=====================================

Phase 4.3 Day 5 Step 6: 用 SWE-bench harness 评估 3 题 generated patches.

按 PROTO-7.19 (grep 真实 schema):
  predictions.jsonl 每行 1 个 JSON object:
    {"instance_id": "...", "model_name_or_path": "r1-distill-14b-lora", 
     "model_patch": "<full patch text>"}

按 PROTO-7.6 (实证, 不脑补):
  swebench harness 用 Docker 隔离 Python version
  → 解决 Day 5 Step 6 fail (Python 3.11 vs 2017 astropy collections.Mapping 问题)
  → Docker images 自动 pull (第一次 ~10-30 min, 后续缓存)

用法 (3 题 batch):
  1. 先准备 predictions.jsonl:
     python scripts/day5_step6_swebench_eval.py --prepare

  2. 跑 harness:
     python -m swebench.harness.run_evaluation \\
       --dataset_name princeton-nlp/SWE-bench_Lite \\
       --predictions_path predictions.jsonl \\
       --max_workers 1 \\
       --run_id day5_3tasks_sanity

  3. 看 results:
     ls logs/run_evaluation/day5_3tasks_sanity/
     cat *.json | python -m json.tool
"""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepare", action="store_true",
                        help="从 3 个 generated_patch.diff 构造 predictions.jsonl")
    parser.add_argument("--candidates", default="day5_candidates.json")
    parser.add_argument("--patches-dir", default="/tmp/swe-bench-day5",
                        help="含 {TASK_ID}/generated_patch.diff 的目录")
    parser.add_argument("--output", default="predictions.jsonl")
    parser.add_argument("--model-name", default="r1-distill-14b-lora-bm25")
    parser.add_argument("--include-tasks", nargs="+",
                        default=None,
                        help="指定 task_ids (默认全 3 题)")
    args = parser.parse_args()

    if not args.prepare:
        print(__doc__)
        return 0

    print("=" * 70)
    print("Day 5 Step 6: 准备 predictions.jsonl")
    print("=" * 70)

    # Load candidates 找 task_ids
    with open(args.candidates) as f:
        candidates = json.load(f)

    # 收集 3 题信息
    tasks_to_eval = []
    for diff_level in ['easy', 'medium', 'hard']:
        t = candidates.get(diff_level)
        if t is None:
            print(f"  ⚠ {diff_level}: candidate is None, skip")
            continue
        instance_id = t['instance_id']
        if args.include_tasks and instance_id not in args.include_tasks:
            continue
        tasks_to_eval.append((diff_level, instance_id))

    print(f"\n  要评估的 task ({len(tasks_to_eval)} 个):")
    for diff_level, tid in tasks_to_eval:
        print(f"    {diff_level}: {tid}")
    print()

    # Read 每个 task 的 generated_patch.diff
    predictions = []
    missing = []
    for diff_level, instance_id in tasks_to_eval:
        patch_path = os.path.join(args.patches_dir, instance_id, "generated_patch.diff")
        if not os.path.exists(patch_path):
            missing.append((diff_level, instance_id, patch_path))
            continue
        
        with open(patch_path, 'r') as f:
            patch_text = f.read()
        
        # 基本验证 (PROTO-7.18 silent failure 警告)
        if 'diff --git' not in patch_text:
            print(f"  ⚠ {instance_id}: patch 不含 'diff --git', 可能不可 apply")
        elif 'diff--git' in patch_text and 'diff --git' not in patch_text:
            print(f"  ❌ {instance_id}: patch 含 'diff--git' (无空格), git 拒绝")
        
        predictions.append({
            "instance_id": instance_id,
            "model_name_or_path": args.model_name,
            "model_patch": patch_text,
        })
        print(f"  ✓ {instance_id}: patch {len(patch_text)} chars loaded")

    if missing:
        print(f"\n  ⚠ Missing patches:")
        for diff_level, tid, p in missing:
            print(f"    {tid}: {p}")

    # Save predictions.jsonl (每行 1 个 JSON)
    print(f"\n[保存] {args.output}")
    with open(args.output, 'w') as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  ✓ {len(predictions)} predictions saved")

    # 打印下一步命令
    instance_ids_str = " ".join(p['instance_id'] for p in predictions)
    
    print()
    print("=" * 70)
    print("下一步: 跑 SWE-bench harness Docker eval")
    print("=" * 70)
    print()
    print("# 选项 1: 完整 batch eval (推荐, 3 题一起)")
    print(f"python -m swebench.harness.run_evaluation \\")
    print(f"    --dataset_name princeton-nlp/SWE-bench_Lite \\")
    print(f"    --predictions_path {args.output} \\")
    print(f"    --max_workers 1 \\")
    print(f"    --run_id day5_3tasks_sanity")
    print()
    print("# 选项 2: 指定 instance_ids (debug 单题)")
    print(f"python -m swebench.harness.run_evaluation \\")
    print(f"    --dataset_name princeton-nlp/SWE-bench_Lite \\")
    print(f"    --predictions_path {args.output} \\")
    print(f"    --instance_ids {instance_ids_str} \\")
    print(f"    --max_workers 1 \\")
    print(f"    --run_id day5_3tasks_sanity")
    print()
    print("# 第一次跑会自动 pull Docker images (3 题需 ~5-10GB, 10-30 min)")
    print("# 之后会复用 cache, 每题 eval 1-3 min")
    print()
    print("# 看 results:")
    print(f"ls logs/run_evaluation/day5_3tasks_sanity/")
    print(f"# Per-task report 在 logs/run_evaluation/{{run_id}}/{{model}}/{{instance_id}}/")
    print(f"# 关键 file: report.json (含 resolved/applied/test_results)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
