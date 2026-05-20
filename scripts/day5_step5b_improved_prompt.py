#!/usr/bin/env python3
"""
scripts/day5_step5b_improved_prompt.py
========================================

Phase 4.3 Fork B (framework v3.6 后): 改进 prompt + post-processing 重跑 3 题.

针对 Day 5 Step 5 发现的 R1 patch format 问题 (T-3.13 SEALED):
  - 67% format 失败 (2/3)
  - 失败模式: diff--git 缺空格 + 散文前缀 + fake hash + markdown fence

改进策略 (T-3.13 反应用):
  1. Prompt 改进:
     - 首行明确: "FIRST LINE OF YOUR RESPONSE MUST BE 'diff --git ...'"
     - 多重负面约束: NO <think>, NO explanation, NO markdown fence
     - 提供 mini-example 模板
  
  2. Post-processing auto-fix:
     - 修复 'diff--git' → 'diff --git' (token bug)
     - 删除 markdown fence ```diff ... ```
     - 删除散文前缀 (任何 'diff --git' 前的非空行)
     - 替换 fake 'index someindex..anotherindex' → 删除该行 (git 不强制)
  
  3. Validation:
     - 调用 `git apply --check` 验证
     - Fail 时 logger.warning (PROTO-7.18) 记录失败模式

PROTO 关联:
  PROTO-7.4 (实测校准): 重跑同 3 题, 看 Fork B 是否改善
  PROTO-7.18 (silent failure 警告): post-process 失败 logger.warning
  PROTO-7.20 (预测前 reverse-search SEALED): 基于 T-3.13/T-3.15 设计

用法 (Fork B 实验):
  python scripts/day5_step5b_improved_prompt.py --task-id astropy__astropy-6938
  python scripts/day5_step5b_improved_prompt.py --task-id django__django-11001
  python scripts/day5_step5b_improved_prompt.py --task-id sympy__sympy-20049

  # 之后 batch eval (覆盖原 predictions.jsonl):
  python scripts/day5_step6_swebench_eval.py --prepare --output predictions_forkB.jsonl
  python -m swebench.harness.run_evaluation \\
      --dataset_name princeton-nlp/SWE-bench_Lite \\
      --predictions_path predictions_forkB.jsonl \\
      --max_workers 1 \\
      --run_id day5_3tasks_forkB
"""

import argparse
import json
import logging
import os
import re
import sys
import time

sys.path.insert(0, os.environ.get('PROJECT_ROOT', '.'))

logger = logging.getLogger(__name__)


# ============================================================================
# Improved prompt template (Fork B)
# ============================================================================

IMPROVED_PROMPT_TEMPLATE = """Fix a bug in {repo} at commit {base_commit_short}.

## Bug Report
{problem_statement}

## Code Context (top-3 BM25 retrieved functions)
{inject_blocks}

## Critical Output Rules

Your response MUST follow these rules:
1. The FIRST CHARACTERS of your response must be "diff --git" (with single space between "diff" and "--git").
2. DO NOT include any explanation, thinking, or commentary before the patch.
3. DO NOT wrap the patch in markdown fence (no ```diff or ``` markers).
4. DO NOT generate fake hashes (no "someindex..anotherindex"). Omit "index XXX..XXX" line entirely.
5. Use real file paths from the retrieved context (e.g. start with "diff --git a/sympy/..." not "a/io/...").
6. End your response immediately after the last line of the patch. No trailing explanation.

## Patch Format Template (MUST match exactly):

diff --git a/path/to/file.py b/path/to/file.py
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -LINE,COUNT +LINE,COUNT @@ optional_context
 unchanged line
-removed line
+added line
 unchanged line

## Your Patch (start immediately, no preamble):

"""


# ============================================================================
# Post-processing auto-fix
# ============================================================================

def fix_patch_format(raw_response: str, repo_name: str = None) -> tuple:
    """
    Auto-fix common R1 patch format errors.
    
    Returns:
        (fixed_patch, fix_log) — fix_log is list of applied fixes
    """
    fix_log = []
    
    # 1. 删除 markdown fence
    if "```" in raw_response:
        # 找 ```diff\n ... ```
        match = re.search(r'```(?:diff)?\s*\n([\s\S]+?)\n```', raw_response)
        if match:
            raw_response = match.group(1)
            fix_log.append("removed_markdown_fence")
    
    # 2. 删除散文前缀 (任何 diff --git 之前的内容)
    diff_start_match = re.search(r'(diff\s*--?-?git\s+a/)', raw_response)
    if not diff_start_match:
        # 也试 'diff--git' (无空格)
        diff_start_match = re.search(r'(diff--git\s+a/)', raw_response)
        if diff_start_match:
            fix_log.append("preserved_diff_dashes_for_fix")
    
    if diff_start_match:
        if diff_start_match.start() > 0:
            raw_response = raw_response[diff_start_match.start():]
            fix_log.append("removed_preamble_prose")
    
    # 3. 修复 'diff--git' → 'diff --git'
    if raw_response.startswith('diff--git'):
        raw_response = 'diff --git' + raw_response[len('diff--git'):]
        fix_log.append("fixed_diff_space")
    
    # 4. 删除 fake 'index someindex..anotherindex' / 'index XXX..XXX' 
    # (但保留真实 hash, 真实 hash 是 7-40 hex 字符)
    lines = raw_response.split('\n')
    fixed_lines = []
    for line in lines:
        # 找 fake index 行
        if re.match(r'^index\s+(someindex|XXX|placeholder)', line, re.IGNORECASE):
            fix_log.append(f"removed_fake_index_line: {line[:50]}")
            continue
        # 找格式像 'index abc123..def456 100644' 但不是合法 hash
        match = re.match(r'^index\s+([a-zA-Z0-9_]+)\.\.([a-zA-Z0-9_]+)(\s+\d+)?$', line)
        if match:
            hash_a, hash_b = match.group(1), match.group(2)
            # 真实 git hash 全部 0-9a-f
            if not (re.match(r'^[0-9a-f]{7,40}$', hash_a) and re.match(r'^[0-9a-f]{7,40}$', hash_b)):
                fix_log.append(f"removed_fake_index_line: {line[:50]}")
                continue
        fixed_lines.append(line)
    raw_response = '\n'.join(fixed_lines)
    
    # 5. 尾部清理 (删除任何 trailing 'This change...' 解释)
    # 找最后 hunk 后的解释
    last_hunk_pos = -1
    for m in re.finditer(r'@@\s+-\d+', raw_response):
        last_hunk_pos = m.start()
    if last_hunk_pos > 0:
        # 从 last_hunk 开始, 找下一个空行 + 非-diff 内容
        post_hunk = raw_response[last_hunk_pos:]
        # Hunk 结束的标志: 一行不以空格/+/-开头 (且不是 \\ No newline...)
        post_lines = post_hunk.split('\n')
        end_idx = len(post_lines)
        for i, line in enumerate(post_lines[1:], 1):  # 跳 @@ 行
            if line and not line.startswith((' ', '+', '-', '@', '\\')):
                # 检查是否新 diff 或 trailing 解释
                if line.startswith('diff --git'):
                    continue  # 下一个 diff
                # 这是 trailing 解释
                end_idx = i
                fix_log.append(f"removed_trailing_prose_from_line_{i}")
                break
        post_hunk = '\n'.join(post_lines[:end_idx])
        raw_response = raw_response[:last_hunk_pos] + post_hunk
    
    return raw_response.strip(), fix_log


# ============================================================================
# Validation
# ============================================================================

def validate_patch_with_git(patch_text: str, repo_path: str) -> tuple:
    """
    用 git apply --check 验证 patch.
    
    Returns:
        (is_valid, error_message)
    """
    import subprocess
    import tempfile
    
    if not patch_text.strip():
        return False, "empty patch"
    
    # 写到 tmp file
    with tempfile.NamedTemporaryFile('w', suffix='.diff', delete=False) as f:
        f.write(patch_text)
        tmp_path = f.name
    
    try:
        result = subprocess.run(
            ['git', '-C', repo_path, 'apply', '--check', tmp_path],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return True, ""
        return False, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "git apply timeout"
    except Exception as e:
        return False, f"git apply error: {e}"
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ============================================================================
# Main pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-id", required=True,
                        help="e.g. astropy__astropy-6938")
    parser.add_argument("--candidates", default="day5_candidates.json")
    parser.add_argument("--patches-dir", default="/tmp/swe-bench-day5")
    parser.add_argument("--difficulty", choices=['easy', 'medium', 'hard'],
                        help="如果 task_id 不在 candidates, 用此指定")
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--validate-with-repo", action="store_true",
                        help="尝试 git apply --check (需 repo 已 clone)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 70)
    print(f"Fork B (Phase 4.3): 改进 prompt + post-fix")
    print(f"Task: {args.task_id}")
    print("=" * 70)

    # === Load candidates ===
    with open(args.candidates) as f:
        candidates = json.load(f)
    
    task = None
    for diff_level, t in candidates.items():
        if t and t.get('instance_id') == args.task_id:
            task = t
            break
    if task is None:
        print(f"❌ Task {args.task_id} not in candidates")
        return 1
    
    # === Load retrieved.json ===
    retrieved_path = os.path.join(args.patches_dir, args.task_id, "retrieved.json")
    if not os.path.exists(retrieved_path):
        print(f"❌ Retrieved not found: {retrieved_path}")
        return 1
    with open(retrieved_path) as f:
        retrieved = json.load(f)
    
    # === Build improved prompt ===
    inject_blocks = "\n\n".join([
        f"### Retrieved Reference {r['rank']}\n```python\n{r['inject']}\n```"
        for r in retrieved
    ])
    
    prompt = IMPROVED_PROMPT_TEMPLATE.format(
        repo=task['repo'],
        base_commit_short=task['base_commit'][:12],
        problem_statement=task['problem_statement'],
        inject_blocks=inject_blocks,
    )
    
    print(f"\n[1] Prompt size: {len(prompt)} chars (~{len(prompt) // 4} tokens)")
    
    # === Generate with R1 ===
    print(f"\n[2] R1 generating (max_new_tokens={args.max_new_tokens})...")
    from knowledge_tree.local_model_clients import make_r1_generator
    
    r1 = make_r1_generator(max_new_tokens=args.max_new_tokens, verbose=args.verbose)
    t0 = time.time()
    response = r1(prompt)
    print(response)
    gen_time = time.time() - t0
    print(f"  ✓ Generated in {gen_time:.1f}s")
    print(f"  Response size: {len(response)} chars")
    r1.unload()
    
    # === Extract + Post-fix ===
    print(f"\n[3] Extracting + auto-fix...")
    # Extract diff block (新 regex, 不依赖 \n\n\n)
    blocks = re.findall(r'(diff\s*--?-?git[\s\S]+?)(?=\ndiff\s*--?-?git|\n```|\Z)', response)
    if not blocks:
        logger.warning("No diff block found in response")
        patch_raw = response.strip()
    else:
        if len(blocks) > 1:
            logger.warning(f"Multiple blocks ({len(blocks)}), selecting longest")
        patch_raw = max(blocks, key=len).strip()
    
    print(f"  Raw extract: {len(patch_raw)} chars")
    
    # Post-fix
    patch_fixed, fix_log = fix_patch_format(patch_raw, repo_name=task['repo'])
    print(f"  After auto-fix: {len(patch_fixed)} chars")
    print(f"  Applied fixes: {fix_log}")
    
    # === Save ===
    output_path = os.path.join(args.patches_dir, args.task_id, "generated_patch_forkB.diff")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(patch_fixed)
    print(f"\n[4] Saved to {output_path}")
    
    # === Validate (optional) ===
    if args.validate_with_repo:
        repo_path = os.path.join(args.patches_dir, args.task_id, "repo")
        if os.path.exists(repo_path):
            print(f"\n[5] Validating with git apply --check...")
            is_valid, error = validate_patch_with_git(patch_fixed, repo_path)
            if is_valid:
                print(f"  ✓ git apply --check PASSED")
            else:
                print(f"  ✗ git apply --check FAILED: {error}")
        else:
            print(f"  ⚠ Repo not found: {repo_path}, skip validation")
    
    # === Save metadata ===
    metadata = {
        'task_id': args.task_id,
        'fork': 'B',
        'prompt_template': 'IMPROVED_PROMPT_TEMPLATE',
        'gen_time_s': round(gen_time, 1),
        'response_size': len(response),
        'raw_extract_size': len(patch_raw),
        'fixed_patch_size': len(patch_fixed),
        'fixes_applied': fix_log,
    }
    metadata_path = os.path.join(args.patches_dir, args.task_id, "forkB_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Metadata: {metadata_path}")
    
    print()
    print("=" * 70)
    print("下一步:")
    print("=" * 70)
    print("  1. 3 题都跑完后, prepare predictions:")
    print(f"     # 改 day5_step6_swebench_eval.py 读 generated_patch_forkB.diff")
    print(f"     # 或手工创建 predictions_forkB.jsonl")
    print()
    print("  2. SWE-bench harness:")
    print(f"     python -m swebench.harness.run_evaluation \\")
    print(f"         --dataset_name princeton-nlp/SWE-bench_Lite \\")
    print(f"         --predictions_path predictions_forkB.jsonl \\")
    print(f"         --max_workers 1 \\")
    print(f"         --run_id day5_3tasks_forkB")
    print()
    print("  3. 看 resolve rate: Fork A 0/3 → Fork B ?/3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
