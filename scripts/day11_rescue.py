#!/usr/bin/env python3
"""
scripts/day11_rescue.py
========================

Phase 4.3 Day 11: 救援机制 (harness unresolved → 离线第二轮).

触发条件 (用户 Q1 选 B): harness 跑完后, 对 apply 成功但 test 失败
(unresolved) 的题做第二轮. 因为主要失败是 "apply 成功但改错位置".

第二轮策略 (基于 Day 9 归因结论修正用户原案):
  - 原案 "LLM 全局树导航筛 top-k": 已证明 R1 做这个净伤害; Claude 可行但本脚本
    用更稳的方式: 扩大 graph candidate (candidate_k 翻倍) + pitfall 负向信号
  - pitfall: 第一轮改了哪个函数 (从 anchor_metadata 的 final_pairs 抽) + 改动 diff,
    作为负向提示 "这个函数/改法已试过, test 没过, 重新考虑"
  - Claude localize (含 pitfall) → 重新选 top-k → regenerate

流程:
  1. 读 harness 结果 (kt-framework-*.json) 的 unresolved_ids
  2. 对每个 unresolved 题:
     a. 读第一轮 patch + anchor_metadata (pitfall 来源)
     b. 重新 retrieve (candidate_k 翻倍, 扩大候选)
     c. Claude localize, prompt 含 pitfall 负向信号
     d. regenerate patch
  3. 输出第二轮 patches 到 rescue work-dir, 供再次 harness eval

用法:
  python scripts/day11_rescue.py \\
      --candidates day10_balanced50.json \\
      --harness-result kt-framework-anchor-based.sonnet.json \\
      --round1-dir /tmp/swe-bench-day10-sonnet \\
      --rescue-dir /tmp/swe-bench-day11-rescue \\
      --model claude_api --claude-model claude-sonnet-4-6 \\
      --candidate-k 30 --top-k 3

  # 限定某个 repo (用户 Q3: 先在一个 repo 验证救援)
  python scripts/day11_rescue.py ... --only-repo sympy
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get('PROJECT_ROOT', '.'))
logger = logging.getLogger(__name__)


# 复用 day7 的 anchor prompt + generation (import)
def _load_day7():
    import importlib.util
    day7_path = Path(__file__).parent / "day7_pipeline.py"
    spec = importlib.util.spec_from_file_location("day7_pipeline", day7_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RESCUE_PITFALL_TEMPLATE = """

## IMPORTANT: Previous attempt failed
A previous fix attempt was made but did NOT pass the tests. Details:

Functions modified in the failed attempt: {failed_functions}

The failed patch was:
```diff
{failed_diff}
```

This means the bug is likely NOT (only) in those functions, OR the fix approach was wrong.
Reconsider carefully:
- Look at OTHER candidate functions, especially ones the failed attempt did not touch.
- The real fix may be in a helper, a caller, or a sibling method.
- Do not simply repeat the same change to the same function.
"""


def extract_failed_functions(metadata_path: Path) -> list[str]:
    """从第一轮 anchor_metadata 抽改了哪些函数."""
    if not metadata_path.exists():
        return []
    try:
        md = json.load(open(metadata_path))
    except Exception:
        return []
    funcs = []
    for p in md.get('final_pairs', []):
        rb = p.get('raw_before', '') or ''
        m = re.search(r'def\s+(\w+)\s*\(', rb)
        if m and m.group(1) not in funcs:
            funcs.append(m.group(1))
    return funcs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--harness-result", required=True,
                        help="harness 输出 JSON (含 unresolved_ids)")
    parser.add_argument("--round1-dir", required=True,
                        help="第一轮 work-dir (含 patch + metadata + ktf)")
    parser.add_argument("--rescue-dir", default="/tmp/swe-bench-rescue")
    parser.add_argument("--model", default="claude_api",
                        choices=['r1', 'nemotron', 'claude_api'])
    parser.add_argument("--claude-model", default="claude-sonnet-4-6")
    parser.add_argument("--claude-effort", default=None,
                        choices=['low', 'medium', 'high', 'xhigh'])
    parser.add_argument("--retriever", default="graph_expanded")
    parser.add_argument("--seed-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=30,
                        help="第二轮扩大候选 (default 30, 第一轮的 2x)")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-expansion", type=int, default=20)
    parser.add_argument("--retry", type=int, default=2)
    parser.add_argument("--only-repo", default=None,
                        help="只救某个 repo (短名, 如 sympy). 验证救援用")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s")

    day7 = _load_day7()

    # 读 unresolved
    harness = json.load(open(args.harness_result))
    unresolved = harness.get('unresolved_ids', [])
    print("=" * 70)
    print(f"Day 11 Rescue: {len(unresolved)} unresolved 题")
    print("=" * 70)

    candidates = json.load(open(args.candidates))
    cand_by_id = {v['instance_id']: v for v in candidates.values()
                  if v and 'instance_id' in v}

    # 过滤 repo
    targets = []
    for tid in unresolved:
        if tid not in cand_by_id:
            logger.warning(f"{tid} not in candidates, skip")
            continue
        if args.only_repo and args.only_repo not in cand_by_id[tid]['repo']:
            continue
        targets.append(tid)

    print(f"救援目标: {len(targets)} 题"
          + (f" (限 repo={args.only_repo})" if args.only_repo else ""))
    for tid in targets:
        print(f"  {tid}")

    if not targets:
        print("无救援目标")
        return 0

    # 加载模型
    if args.model == 'claude_api':
        from knowledge_tree.claude_api_client import ClaudeAPICallable
        model_callable = ClaudeAPICallable(
            model=args.claude_model, effort=args.claude_effort, verbose=args.verbose)
    elif args.model == 'r1':
        from knowledge_tree.local_model_clients import make_r1_generator
        model_callable = make_r1_generator(verbose=args.verbose)
    else:
        from knowledge_tree.local_model_clients import LocalModelCallable
        model_callable = LocalModelCallable(base_model="./models/nemotron-nano-9b-v2",
                                            use_int4=True, verbose=args.verbose)

    rescue_dir = Path(args.rescue_dir)
    rescue_dir.mkdir(parents=True, exist_ok=True)
    round1_dir = Path(args.round1_dir)

    from knowledge_tree.storage import JSONStorage
    from knowledge_tree.core import KnowledgeTree
    from knowledge_tree.retrievers import GraphExpandedRetriever
    from knowledge_tree.localizer import localize, reorder_by_localization

    summary = []
    for i, tid in enumerate(targets):
        task = cand_by_id[tid]
        r1_task_dir = round1_dir / tid
        rescue_task_dir = rescue_dir / tid
        rescue_task_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{i+1}/{len(targets)}] 救援 {tid}")

        # pitfall: 第一轮改了哪些函数 + diff
        failed_funcs = extract_failed_functions(r1_task_dir / "anchor_metadata.json")
        failed_diff = ""
        r1_patch = r1_task_dir / "generated_patch.diff"
        if r1_patch.exists():
            failed_diff = r1_patch.read_text()[:1500]  # 截断避免过长
        print(f"  第一轮改了: {failed_funcs}")

        # KTF (复用第一轮)
        ktf_path = r1_task_dir / "ktf.json"
        if not ktf_path.exists():
            print(f"  ✗ KTF 不存在 {ktf_path}, skip")
            summary.append({'task_id': tid, 'status': 'no_ktf'})
            continue

        repo_path = r1_task_dir / "repo"
        if not repo_path.exists():
            print(f"  ✗ repo 不存在, skip")
            summary.append({'task_id': tid, 'status': 'no_repo'})
            continue

        storage = JSONStorage(str(ktf_path), create_if_missing=False)
        tree = KnowledgeTree(storage.list_all())

        # 第二轮 retrieve: 扩大 candidate
        retriever = GraphExpandedRetriever(tree, seed_k=args.seed_k,
                                            max_expansion=args.max_expansion)
        retrieved = retriever.retrieve(task['problem_statement'], top_k=args.candidate_k)
        print(f"  第二轮召回 {len(retrieved)} 候选 (candidate_k={args.candidate_k})")

        # localize with pitfall
        pitfall = RESCUE_PITFALL_TEMPLATE.format(
            failed_functions=", ".join(failed_funcs) or "(unknown)",
            failed_diff=failed_diff or "(empty)",
        )
        # localize 的 problem_statement 附加 pitfall
        loc_problem = task['problem_statement'] + pitfall
        loc_result = localize(loc_problem, retrieved, model_callable, select_k=args.top_k)
        print(f"  localize 选中: {loc_result.selected_ids}")
        retrieved = reorder_by_localization(retrieved, loc_result)[:args.top_k]

        # regenerate (用 day7 的 generation, prompt 也带 pitfall)
        # 构造带 pitfall 的 generation: 复用 day7 step_generate_and_synth
        # 但 problem_statement 加 pitfall
        task_with_pitfall = dict(task)
        task_with_pitfall['problem_statement'] = task['problem_statement'] + pitfall

        gen_md = day7.step_generate_and_synth(
            task_with_pitfall, retrieved, repo_path, rescue_task_dir,
            model_callable, args.model, args.top_k, args.retry,
        )
        status = 'rescued_apply_ok' if gen_md['final_git_apply_ok'] else 'apply_failed'
        print(f"  第二轮: {status}, patch={gen_md['final_patch_size']} chars")
        summary.append({
            'task_id': tid, 'status': status,
            'failed_functions_round1': failed_funcs,
            'selected_round2': loc_result.selected_ids,
            'apply_ok': gen_md['final_git_apply_ok'],
        })

    if hasattr(model_callable, 'unload'):
        model_callable.unload()

    # 保存救援 summary
    (rescue_dir / "rescue_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    apply_ok = sum(1 for s in summary if s.get('apply_ok'))
    print(f"\n{'='*70}")
    print(f"救援完成: {len(summary)} 题, {apply_ok} apply OK")
    print(f"第二轮 patches: {rescue_dir}")
    print(f"\n下一步: 对救援 patches 再跑 harness")
    print(f"  python scripts/day7_step6_prepare.py --candidates {args.candidates} \\")
    print(f"      --patches-dir {rescue_dir} --output predictions_rescue.jsonl")
    print(f"  python -m swebench.harness.run_evaluation \\")
    print(f"      --dataset_name princeton-nlp/SWE-bench_Lite \\")
    print(f"      --predictions_path predictions_rescue.jsonl --run_id rescue")
    return 0


if __name__ == "__main__":
    sys.exit(main())
