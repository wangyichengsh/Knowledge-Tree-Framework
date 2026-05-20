#!/usr/bin/env python3
"""
scripts/day6_step5_anchor_generate.py
=======================================

Phase 4.3 Day 6 Step 5: Anchor-based generation pipeline.

替代 day5_step5_generate.py 中"R1 直接输出 unified diff"的 naive 方式.

新流程:
  1. R1 输出 BEFORE/AFTER pairs (语义信息)
  2. 程序在 retrieved nodes' source_code + 真实文件中搜索 BEFORE
  3. difflib + 真实 line numbers 合成合法 unified diff
  4. 自动 git apply --check 验证 (PROTO-7.27)
  5. 自动 trailing newline (SWE-bench Issue #145 教训)

设计动机 (T-3.13 v1.2 边界层 + 语义层修复, T-3.14 v1.2 R1 counting failure):
  实证 (Day 6 Step 1): 修复 source_code 后 R1 仍 line number 错 (335 vs 真实 356).
  → counting + indentation 是 LLM 内在弱点, 必须 program-assist.

用法:
  python scripts/day6_step5_anchor_generate.py \\
      --task-id django__django-11001 \\
      --difficulty medium

输出:
  /tmp/swe-bench-day5/{task_id}/generated_patch.diff   (合法 unified diff)
  /tmp/swe-bench-day5/{task_id}/anchor_metadata.json   (诊断信息)
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.environ.get('PROJECT_ROOT', '.'))

logger = logging.getLogger(__name__)


# ============================================================================
# Anchor-based prompt template
# ============================================================================

ANCHOR_PROMPT_TEMPLATE = """You are a senior Python engineer fixing a bug in {repo}.

## Bug Report
{problem_statement}

## Code Context (top-{top_k} BM25 retrieved functions, with full source code)
{inject_blocks}

## Output Format (STRICT)

Output your fix as one or more CHANGE blocks. After your reasoning, output the blocks exactly like the EXAMPLE below.

## EXAMPLE (study this format carefully)

Suppose the bug is "func returns None when x is 0". You would output:

REASONING: The function returns None when input is 0 because the `if x:` check is falsy for 0. Fix: change to `if x is not None:`.

CHANGE 1:
BEFORE:
```python
    if x:
        return x * 2
    return None
```
AFTER:
```python
    if x is not None:
        return x * 2
    return None
```

(End of example.)

## CRITICAL RULES

1. BEFORE must be COPIED VERBATIM from the source code in "Code Context" above.
   This means: same indentation (leading spaces), same line breaks, same characters.
   DO NOT strip indentation. DO NOT paraphrase. DO NOT use `# BEFORE` style comments.
2. Use `BEFORE:` and `AFTER:` as section labels (NOT `# BEFORE` comments inside code).
3. Each CHANGE block needs its own BEFORE and AFTER (separate code fences).
4. BEFORE should be 2-5 lines (enough to uniquely identify the location).
5. AFTER must use the SAME indentation as BEFORE (so the patch applies cleanly).
6. If multiple sites need similar fixes, output multiple CHANGE blocks (CHANGE 1, CHANGE 2, ...).
7. Do NOT output a unified diff. Do NOT use `@@` markers. Do NOT use `diff --git`.
8. Do NOT output line numbers. The program will compute them from BEFORE.
9. For Python string literals, use proper escaping: `'\\n'` for newline, not `'n'`.

## Your Output (start with "REASONING:" then CHANGE blocks):

"""


# ============================================================================
# Main pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-id", required=True,
                        help="e.g. django__django-11001")
    parser.add_argument("--candidates", default="day5_candidates.json")
    parser.add_argument("--patches-dir", default="/tmp/swe-bench-day5")
    parser.add_argument("--difficulty", choices=['easy', 'medium', 'hard'],
                        required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=3,
                        help="多少 retrieved nodes 喂 R1 (默认 3)")
    parser.add_argument("--validate-with-repo", action="store_true",
                        default=True,
                        help="git apply --check 验证")
    parser.add_argument("--retry-on-fail", type=int, default=1,
                        help="如果首次合成 fail, 重试次数 (反馈给 R1)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 70)
    print(f"Day 6 Anchor-based generation")
    print(f"Task: {args.task_id} ({args.difficulty})")
    print("=" * 70)

    # === Load task ===
    candidates = json.load(open(args.candidates))
    task = candidates.get(args.difficulty)
    if not task or task['instance_id'] != args.task_id:
        # search all
        for diff_level, t in candidates.items():
            if t and t.get('instance_id') == args.task_id:
                task = t
                break
    if not task:
        print(f"❌ Task {args.task_id} not in candidates")
        return 1

    # === Load retrieved (with source_code in inject) ===
    task_dir = Path(args.patches_dir) / args.task_id
    retrieved_path = task_dir / "retrieved.json"
    if not retrieved_path.exists():
        print(f"❌ Retrieved not found: {retrieved_path}")
        print(f"   Run day5_step3_retrieve.py first.")
        return 1
    retrieved_raw = json.load(open(retrieved_path))
    
    # === Reconstruct KnowledgeNode objects (for anchor_diff.locate_anchor) ===
    # retrieved.json 中保存的是 {rank, id, inject} 字典格式
    # 需要从 ktf.json 重读 node 拿 source_code 和 domain_metadata
    ktf_path = task_dir / "ktf.json"
    if not ktf_path.exists():
        print(f"❌ KTF not found: {ktf_path}")
        return 1
    
    from knowledge_tree.storage import JSONStorage
    from knowledge_tree.core import KnowledgeNode
    storage = JSONStorage(str(ktf_path), create_if_missing=False)
    all_nodes_dict = {n.id: n for n in storage.list_all()}
    
    retrieved_node_ids = [r['id'] for r in retrieved_raw[:args.top_k]]
    retrieved_nodes = [all_nodes_dict[nid] for nid in retrieved_node_ids if nid in all_nodes_dict]
    
    print(f"\n[1] Retrieved {len(retrieved_nodes)} nodes with source_code")
    for r, n in zip(retrieved_raw[:args.top_k], retrieved_nodes):
        has_sc = bool(getattr(n, 'source_code', None))
        sc_len = len(n.source_code) if has_sc else 0
        print(f"    rank {r['rank']}: {n.id}")
        print(f"        file: {n.domain_metadata.get('file', 'N/A')}")
        print(f"        source_code: {'YES' if has_sc else 'NO'} ({sc_len} chars)")

    # === Build prompt ===
    inject_blocks = "\n\n".join([
        f"### Retrieved Code Reference {r['rank']}: {n.domain_metadata.get('qualified_name', n.id)}\n"
        f"File: `{n.domain_metadata.get('file', 'unknown')}`\n"
        f"```python\n{n.source_code if n.source_code else '(no source code)'}\n```"
        for r, n in zip(retrieved_raw[:args.top_k], retrieved_nodes)
    ])
    
    prompt = ANCHOR_PROMPT_TEMPLATE.format(
        repo=task['repo'],
        problem_statement=task['problem_statement'],
        top_k=args.top_k,
        inject_blocks=inject_blocks,
    )
    print(f"\n[2] Prompt size: {len(prompt)} chars (~{len(prompt) // 4} tokens)")

    # === Repo path (for anchor search + git apply) ===
    repo_root = task_dir / "repo"
    if not repo_root.exists():
        print(f"❌ Repo not found: {repo_root}")
        print(f"   Run day5_step1 (clone repo) first.")
        return 1

    # === Generate (with retry on fail) ===
    from knowledge_tree.local_model_clients import make_r1_generator, make_nemotron_retriever
    from knowledge_tree.anchor_diff import response_to_unified_diff
    
    r1 = make_r1_generator(max_new_tokens=args.max_new_tokens, verbose=args.verbose)
    # r1 = make_nemotron_retriever(max_new_tokens=args.max_new_tokens, verbose=args.verbose)
    
    final_patch = ""
    final_pairs = []
    final_warnings = []
    attempt_log = []
    
    current_prompt = prompt
    for attempt in range(args.retry_on_fail + 1):
        print(f"\n[3.{attempt}] R1 generating (attempt {attempt + 1}/{args.retry_on_fail + 1})...")
        t0 = time.time()
        response = r1(current_prompt)
        print(response)
        gen_time = time.time() - t0
        print(f"    ✓ Generated in {gen_time:.1f}s ({len(response)} chars)")
        
        # === Parse + synth ===
        print(f"[4.{attempt}] Parsing BEFORE/AFTER + synthesizing diff...")
        patch_text, pairs, warnings = response_to_unified_diff(
            response, retrieved_nodes, repo_root,
        )
        
        print(f"    Anchored pairs: {sum(1 for p in pairs if p.match_status != 'not_found')}/{len(pairs)}")
        for p in pairs:
            status_marker = "✓" if p.match_status != 'not_found' else "✗"
            print(f"      {status_marker} [{p.match_status}] {p.raw_before[:60]!r}...")
        if warnings:
            print(f"    Warnings:")
            for w in warnings:
                print(f"      - {w}")
        
        attempt_log.append({
            'attempt': attempt + 1,
            'response_size': len(response),
            'gen_time': round(gen_time, 1),
            'pairs_found': len(pairs),
            'pairs_anchored': sum(1 for p in pairs if p.match_status != 'not_found'),
            'patch_size': len(patch_text),
            'warnings': warnings,
        })
        
        # === git apply --check ===
        if patch_text and args.validate_with_repo:
            print(f"[5.{attempt}] git apply --check...")
            patch_path_tmp = task_dir / "anchor_patch_tmp.diff"
            patch_path_tmp.write_text(patch_text)
            result = subprocess.run(
                ['git', '-C', str(repo_root), 'apply', '--check', str(patch_path_tmp.resolve())],
                capture_output=True, text=True,
            )
            patch_path_tmp.unlink(missing_ok=True)
            if result.returncode == 0:
                print(f"    ✓ git apply --check PASSED")
                final_patch = patch_text
                final_pairs = pairs
                final_warnings = warnings
                break
            else:
                print(f"    ✗ git apply --check FAILED: {result.stderr.strip()}")
                attempt_log[-1]['git_apply_error'] = result.stderr.strip()
                if attempt < args.retry_on_fail:
                    # 构造 retry prompt
                    current_prompt = prompt + f"""

PREVIOUS ATTEMPT FAILED with: {result.stderr.strip()}

Please try again. Pay extra attention to:
- BEFORE must be COPIED VERBATIM from the source code in "Code Context" above
- Multiple consecutive lines OK
- Use real existing code, not paraphrased

## Your Output:
"""
                    continue
        else:
            # 无 validation, 直接接受
            final_patch = patch_text
            final_pairs = pairs
            final_warnings = warnings
            if not patch_text:
                print(f"    ⚠ empty patch")
            break

    r1.unload()

    # === Save outputs ===
    patch_path = task_dir / "generated_patch.diff"
    if final_patch:
        patch_path.write_text(final_patch)
        print(f"\n[6] Saved patch to {patch_path}")
        print(f"    Size: {len(final_patch)} chars")
    else:
        # 即使空也写一个文件 (避免 step6 missing)
        patch_path.write_text("")
        print(f"\n[6] ⚠ No valid patch generated. Empty file saved to {patch_path}")

    # Metadata for diagnosis
    metadata_path = task_dir / "anchor_metadata.json"
    metadata = {
        'task_id': args.task_id,
        'fork': 'anchor-based',
        'top_k': args.top_k,
        'attempts': attempt_log,
        'final_pairs': [
            {
                'raw_before': p.raw_before[:200],
                'match_status': p.match_status,
                'matched_file': p.matched_file,
                'matched_start_line': p.matched_start_line,
            }
            for p in final_pairs
        ],
        'final_warnings': final_warnings,
        'final_patch_size': len(final_patch),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"    Metadata: {metadata_path}")

    print()
    print("=" * 70)
    print("下一步:")
    print("=" * 70)
    print(f"  1. (可选) 看其他 2 题: 改 --task-id + --difficulty 重跑")
    print(f"  2. 准备 predictions.jsonl:")
    print(f"     python scripts/day5_step6_swebench_eval.py --prepare")
    print(f"  3. SWE-bench harness:")
    print(f"     python -m swebench.harness.run_evaluation \\")
    print(f"         --dataset_name princeton-nlp/SWE-bench_Lite \\")
    print(f"         --predictions_path predictions.jsonl \\")
    print(f"         --run_id day6_anchor_3tasks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
