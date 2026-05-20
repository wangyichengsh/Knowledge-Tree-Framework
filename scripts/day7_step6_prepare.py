#!/usr/bin/env python3
"""
scripts/day7_step6_prepare.py
==============================

Phase 4.3 Day 7: 准备 predictions.jsonl 给 SWE-bench harness.

与 day5_step6 区别:
  - candidates keys 可以是任意 (e.g. task_0, task_1, ...) 而非固定 easy/medium/hard
  - 自动遍历所有 keys
  - 默认 patches-dir 是 /tmp/swe-bench-day7

用法:
  python scripts/day7_step6_prepare.py
  python scripts/day7_step6_prepare.py --candidates day7_candidates.json --output predictions_day7.jsonl

然后:
  python -m swebench.harness.run_evaluation \\
      --dataset_name princeton-nlp/SWE-bench_Lite \\
      --predictions_path predictions_day7.jsonl \\
      --max_workers 4 \\
      --run_id day7_pilot
"""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", default="day7_candidates.json",
                        help="Day7 candidates (任意 keys)")
    parser.add_argument("--patches-dir", default="/tmp/swe-bench-day7",
                        help="含 {task_id}/generated_patch.diff 的目录")
    parser.add_argument("--output", default="predictions_day7.jsonl")
    parser.add_argument("--model-name", default="kt-framework-anchor-based",
                        help="模型 ID (会写入 predictions, 用于 harness 区分)")
    parser.add_argument("--skip-empty", action="store_true", default=True,
                        help="跳过空 patch (默认 True)")
    args = parser.parse_args()

    print("=" * 70)
    print("Day 7 Step 6: 准备 predictions.jsonl")
    print("=" * 70)

    with open(args.candidates) as f:
        candidates = json.load(f)

    # 遍历所有 candidates (任意 keys)
    predictions = []
    missing = []
    empty_skipped = []

    for key, t in candidates.items():
        if t is None or 'instance_id' not in t:
            continue
        instance_id = t['instance_id']
        patch_path = os.path.join(args.patches_dir, instance_id, "generated_patch.diff")

        if not os.path.exists(patch_path):
            missing.append((key, instance_id, patch_path))
            continue

        with open(patch_path) as f:
            patch_text = f.read()

        # Skip empty
        if args.skip_empty and not patch_text.strip():
            empty_skipped.append((key, instance_id))
            continue

        # Sanity
        if 'diff --git' not in patch_text:
            print(f"  ⚠ {instance_id}: 不含 'diff --git'")

        predictions.append({
            "instance_id": instance_id,
            "model_name_or_path": args.model_name,
            "model_patch": patch_text,
        })
        diff = t.get('difficulty', '?')
        print(f"  ✓ [{diff}] {instance_id}: patch {len(patch_text)} chars")

    if missing:
        print(f"\n⚠ Missing patches ({len(missing)}):")
        for key, tid, p in missing:
            print(f"    {tid}: {p}")

    if empty_skipped:
        print(f"\n⚠ Empty patches skipped ({len(empty_skipped)}):")
        for key, tid in empty_skipped:
            print(f"    {tid}")

    # Save
    with open(args.output, 'w') as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\n✓ {len(predictions)} predictions saved to {args.output}")

    # 下一步
    if predictions:
        instance_ids = [p['instance_id'] for p in predictions]
        print()
        print("=" * 70)
        print("下一步: 跑 SWE-bench harness")
        print("=" * 70)
        print()
        print(f"# 全部 {len(predictions)} 题")
        print(f"python -m swebench.harness.run_evaluation \\")
        print(f"    --dataset_name princeton-nlp/SWE-bench_Lite \\")
        print(f"    --predictions_path {args.output} \\")
        print(f"    --max_workers 4 \\")
        print(f"    --run_id day7_pilot")
        print()
        print(f"# 看结果:")
        print(f"ls logs/run_evaluation/day7_pilot/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
