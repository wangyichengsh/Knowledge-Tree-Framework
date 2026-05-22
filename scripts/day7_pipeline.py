#!/usr/bin/env python3
"""
scripts/day7_pipeline.py
=========================

Phase 4.3 Day 7: 统一 SWE-bench task 端到端 pipeline.

替代 Day 5/6 的手工 6 步 (select → clone → build → retrieve → generate → eval),
一条命令跑完任意题数 + 任意模型. 中间文件保留供分析.

设计目标:
  - 多 task 批处理 (--tasks 指定, 或 --select-from-lite 自动选)
  - 参数化模型 (--model r1 | nemotron)
  - 修复 retry bug (empty patch 也重试, 不 silently break)
  - 中间文件保留 (clone repo, KTF, retrieved.json, raw response, patch, metadata)
  - 输出汇总 (apply rate, 各题 status 分布)

依赖:
  - knowledge_tree.ast_tree_builder (path_prefix 支持)
  - knowledge_tree.anchor_diff (4 格式 + fuzzy + real-file fallback)
  - knowledge_tree.local_model_clients (make_r1_generator / make_nemotron_retriever)
  - datasets (princeton-nlp/SWE-bench_Lite)

用法:
  # 单题
  python scripts/day7_pipeline.py --tasks django__django-11001 --model r1

  # 多题 (从 SWE-bench Lite 选)
  python scripts/day7_pipeline.py --select-easy 3 --select-medium 5 --select-hard 2 --model r1

  # 已有 candidates.json 直接跑
  python scripts/day7_pipeline.py --candidates day5_candidates.json --model nemotron

  # 跳过 build (KTF 已存在), 仅重 retrieve + generate
  python scripts/day7_pipeline.py --tasks django__django-11001 --skip-build --model r1

  # 控制 retry / top-k
  python scripts/day7_pipeline.py --tasks ... --top-k 5 --retry 2

输出:
  /tmp/swe-bench-day7/
    <task_id>/
      repo/                              (git clone)
      ktf.json                            (ASTTreeBuilder 输出)
      retrieved.json                      (top-k 节点 metadata)
      raw_response_attempt_N.txt          (R1/Nemotron 原始输出, 每次 attempt 一个)
      generated_patch.diff                (合成的 unified diff)
      anchor_metadata.json                (诊断信息)
    day7_summary.json                     (全部 task 汇总)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.environ.get('PROJECT_ROOT', '.'))

logger = logging.getLogger(__name__)


# ============================================================================
# Anchor-based prompt template (与 day6_step5 一致)
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
10. NOTE: The bug report may contain illustrative reproducer code (e.g. test snippets).
    Do NOT modify reproducer code. Look for the actual implementation in retrieved code.

## Your Output (start with "REASONING:" then CHANGE blocks):

"""


# ============================================================================
# Task selection (from SWE-bench Lite)
# ============================================================================

def classify_task_difficulty(t: dict) -> str:
    """复制 day5_select_3_tasks 分类逻辑."""
    ps = t.get('patch_size', 0)
    prs = t.get('problem_size', 0)
    ftp = t.get('n_fail_to_pass', 0)
    
    if ps <= 500 and prs <= 2000 and 1 <= ftp <= 2:
        return 'easy'
    elif 500 < ps <= 1500 and 1500 <= prs <= 3000 and 1 <= ftp <= 3:
        return 'medium'
    elif ps > 1500 and prs > 2000 and ftp >= 2:
        return 'hard'
    return 'other'


def load_swebench_lite_tasks() -> list[dict]:
    """加载 SWE-bench Lite 全部 task + 计算 metrics."""
    from datasets import load_dataset
    ds = load_dataset('princeton-nlp/SWE-bench_Lite', split='test')
    tasks = list(ds)
    for t in tasks:
        t['patch_size'] = len(t.get('patch', ''))
        t['test_patch_size'] = len(t.get('test_patch', ''))
        t['problem_size'] = len(t.get('problem_statement', ''))
        fail_str = t.get('FAIL_TO_PASS', '[]')
        try:
            t['n_fail_to_pass'] = len(json.loads(fail_str))
        except Exception:
            t['n_fail_to_pass'] = 0
    return tasks


def select_tasks(
    n_easy: int = 0, n_medium: int = 0, n_hard: int = 0,
    repos: Optional[list] = None,
) -> list[dict]:
    """按难度 + repo 选 task."""
    tasks = load_swebench_lite_tasks()
    
    by_diff = defaultdict(list)
    for t in tasks:
        diff = classify_task_difficulty(t)
        if diff in ('easy', 'medium', 'hard'):
            if repos and t['repo'] not in repos:
                continue
            by_diff[diff].append(t)
    
    selected = []
    selected.extend(by_diff['easy'][:n_easy])
    selected.extend(by_diff['medium'][:n_medium])
    selected.extend(by_diff['hard'][:n_hard])
    return selected


def filter_tasks_by_ids(task_ids: list[str]) -> list[dict]:
    """按 instance_id 列表选 task."""
    tasks = load_swebench_lite_tasks()
    by_id = {t['instance_id']: t for t in tasks}
    result = []
    for tid in task_ids:
        if tid in by_id:
            result.append(by_id[tid])
        else:
            logger.warning(f"Task ID not in SWE-bench Lite: {tid}")
    return result


def task_to_record(t: dict) -> dict:
    """转 task 为最小记录 (避免 json 序列化大 fields)."""
    return {
        'instance_id': t['instance_id'],
        'repo': t['repo'],
        'base_commit': t['base_commit'],
        'patch_size': t['patch_size'],
        'problem_size': t['problem_size'],
        'n_fail_to_pass': t['n_fail_to_pass'],
        'problem_statement': t['problem_statement'],
        'patch': t['patch'],
        'test_patch': t['test_patch'],
        'FAIL_TO_PASS': t.get('FAIL_TO_PASS', '[]'),
        'PASS_TO_PASS': t.get('PASS_TO_PASS', '[]'),
        'environment_setup_commit': t.get('environment_setup_commit', ''),
        'version': t.get('version', ''),
        'difficulty': classify_task_difficulty(t),
    }


# ============================================================================
# Per-task steps
# ============================================================================

def step_clone_repo(task: dict, work_dir: Path) -> Path:
    """Step 1: clone repo + checkout base_commit."""
    repo_url = f"https://github.com/{task['repo']}"
    base_commit = task['base_commit']
    repo_path = work_dir / "repo"
    
    if repo_path.exists() and (repo_path / '.git').exists():
        # 已 clone, 检查 commit
        try:
            current = subprocess.run(
                ['git', '-C', str(repo_path), 'rev-parse', 'HEAD'],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            if current.startswith(base_commit[:12]):
                logger.info(f"  Repo already at base_commit {base_commit[:12]}, skip clone")
                return repo_path
        except subprocess.CalledProcessError:
            pass
        # Commit 不对, 重新 checkout
        try:
            subprocess.run(
                ['git', '-C', str(repo_path), 'reset', '--hard', base_commit],
                capture_output=True, text=True, check=True,
            )
            logger.info(f"  Reset existing repo to {base_commit[:12]}")
            return repo_path
        except subprocess.CalledProcessError as e:
            logger.warning(f"  Reset failed, will re-clone: {e.stderr}")
            subprocess.run(['rm', '-rf', str(repo_path)])
    
    logger.info(f"  Cloning {repo_url}...")
    subprocess.run(
        ['git', 'clone', '--quiet', repo_url, str(repo_path)],
        check=True,
    )
    subprocess.run(
        ['git', '-C', str(repo_path), 'checkout', '--quiet', base_commit],
        check=True,
    )
    return repo_path


def step_build_ktf(task: dict, repo_path: Path, ktf_path: Path, skip_if_exists: bool = True) -> int:
    """Step 2: ASTTreeBuilder build KTF.
    
    Returns: 节点数
    """
    if skip_if_exists and ktf_path.exists():
        logger.info(f"  KTF already exists: {ktf_path}, skip build")
        from knowledge_tree.storage import JSONStorage
        storage = JSONStorage(str(ktf_path), create_if_missing=False)
        return len(storage.list_all())
    
    from knowledge_tree.ast_tree_builder import ASTTreeBuilder
    from knowledge_tree.storage import JSONStorage
    
    # 决定 src_dir + path_prefix (Phase 4.3 Day 6 path_prefix fix)
    package_name = task['repo'].split('/')[1]
    src_dir = repo_path / package_name
    if not src_dir.exists():
        src_dir = repo_path
        path_prefix = None
        logger.info(f"  Build full repo (no package subdir)")
    else:
        path_prefix = package_name
        logger.info(f"  Build subdir {package_name}/ with path_prefix='{package_name}'")
    
    builder = ASTTreeBuilder(
        include_classes=True,
        include_sub_functions=False,
        path_prefix=path_prefix,
    )
    t_start = time.time()
    nodes = builder.build_from_repo(
        repo_path=str(src_dir),
        file_glob="**/*.py",
        ignore_patterns=['tests/', 'test_*.py', 'docs/', '__pycache__', '.tox/'],
    )
    build_time = time.time() - t_start
    logger.info(f"  Built {len(nodes)} nodes in {build_time:.1f}s, stats: {builder.get_stats()}")
    
    # Sanity: 第一个 node 的 file path 应含 prefix
    if path_prefix and nodes:
        sample_file = nodes[0].domain_metadata.get('file', '')
        if not sample_file.startswith(path_prefix):
            logger.warning(f"  ⚠ sample path '{sample_file}' missing prefix '{path_prefix}'")
    
    # Save
    ktf_path.parent.mkdir(parents=True, exist_ok=True)
    storage = JSONStorage(str(ktf_path), create_if_missing=True, autosave=False)
    for n in nodes:
        storage.save_node(n)
    storage.flush()
    logger.info(f"  Saved to {ktf_path}")
    return len(nodes)


def step_retrieve(
    task: dict, ktf_path: Path, retrieved_path: Path, top_k: int = 3,
) -> tuple[list, dict]:
    """Step 3: BM25 retrieve top-k.
    
    Returns:
        (retrieved_nodes, retrieve_stats)
    """
    from knowledge_tree.storage import JSONStorage
    from knowledge_tree.core import KnowledgeTree
    from knowledge_tree.retrievers import BM25Retriever
    
    storage = JSONStorage(str(ktf_path), create_if_missing=False)
    nodes_list = storage.list_all()
    tree = KnowledgeTree(nodes_list)
    logger.info(f"  Loaded {len(tree)} nodes from KTF")
    
    bm25 = BM25Retriever(tree)
    retrieved = bm25.retrieve(task['problem_statement'], top_k=top_k)
    
    # Oracle file 命中检查
    oracle_files = re.findall(r'diff --git a/(\S+)', task['patch'])
    hits = sum(
        1 for n in retrieved
        if any(of in n.domain_metadata.get('file', '') for of in oracle_files)
    )
    logger.info(f"  Top-{top_k}: {[n.id for n in retrieved]}")
    logger.info(f"  Oracle file hits: {hits}/{top_k} (oracle_files={oracle_files})")
    
    # Save retrieved.json (与 day5_step3 兼容)
    retrieved_data = [{
        'rank': i + 1,
        'id': n.id,
        'inject': n.llm_inject_text(),
    } for i, n in enumerate(retrieved)]
    retrieved_path.write_text(json.dumps(retrieved_data, indent=2, ensure_ascii=False))
    
    return retrieved, {
        'top_k': top_k,
        'oracle_files': oracle_files,
        'oracle_hits': hits,
        'retrieved_ids': [n.id for n in retrieved],
    }


def step_generate_and_synth(
    task: dict, retrieved_nodes: list, repo_root: Path, task_dir: Path,
    model_callable, model_name: str, top_k: int = 3,
    max_retries: int = 2,
) -> dict:
    """Step 5: Anchor-based generation + synth + git apply check.
    
    Returns: 完整诊断 dict
    """
    from knowledge_tree.anchor_diff import response_to_unified_diff
    
    # Build prompt
    inject_blocks = "\n\n".join([
        f"### Retrieved Code Reference {i+1}: {n.domain_metadata.get('qualified_name', n.id)}\n"
        f"File: `{n.domain_metadata.get('file', 'unknown')}`\n"
        f"```python\n{n.source_code if n.source_code else '(no source code)'}\n```"
        for i, n in enumerate(retrieved_nodes)
    ])
    prompt = ANCHOR_PROMPT_TEMPLATE.format(
        repo=task['repo'],
        problem_statement=task['problem_statement'],
        top_k=top_k,
        inject_blocks=inject_blocks,
    )
    logger.info(f"  Prompt size: {len(prompt)} chars (~{len(prompt)//4} tokens)")
    
    # Retry loop (FIX: empty patch 也重试, 不 silently break)
    final_patch = ""
    final_pairs = []
    final_warnings = []
    attempt_log = []
    
    current_prompt = prompt
    for attempt_idx in range(max_retries + 1):
        attempt_label = f"attempt {attempt_idx + 1}/{max_retries + 1}"
        logger.info(f"  [{model_name}] Generating ({attempt_label})...")
        
        t0 = time.time()
        response = model_callable(current_prompt)
        gen_time = time.time() - t0
        logger.info(f"    Generated in {gen_time:.1f}s ({len(response)} chars)")
        
        # 保存 raw response (诊断用)
        (task_dir / f"raw_response_attempt_{attempt_idx + 1}.txt").write_text(response)
        
        # Parse + synth
        patch_text, pairs, warnings = response_to_unified_diff(
            response, retrieved_nodes, repo_root,
        )
        anchored = sum(1 for p in pairs if p.match_status != 'not_found')
        logger.info(f"    Parsed: {len(pairs)} pairs, anchored: {anchored}")
        for p in pairs:
            marker = "✓" if p.match_status != 'not_found' else "✗"
            logger.info(f"      {marker} [{p.match_status}] {(p.raw_before or '<empty>')[:60]!r}")
        
        # 验证 patch 是否合法 (git apply --check)
        apply_ok = False
        apply_error = ""
        if patch_text:
            patch_path_tmp = task_dir / f"patch_tmp_attempt_{attempt_idx + 1}.diff"
            patch_path_tmp.write_text(patch_text)
            result = subprocess.run(
                ['git', '-C', str(repo_root), 'apply', '--check', str(patch_path_tmp.resolve())],
                capture_output=True, text=True,
            )
            apply_ok = (result.returncode == 0)
            apply_error = result.stderr.strip() if not apply_ok else ""
            patch_path_tmp.unlink(missing_ok=True)
        else:
            apply_error = "empty patch (no anchored pairs)"
        
        attempt_log.append({
            'attempt': attempt_idx + 1,
            'response_size': len(response),
            'gen_time_s': round(gen_time, 1),
            'pairs_found': len(pairs),
            'pairs_anchored': anchored,
            'pair_statuses': [p.match_status for p in pairs],
            'patch_size': len(patch_text),
            'git_apply_ok': apply_ok,
            'git_apply_error': apply_error,
            'warnings': warnings,
        })
        
        if apply_ok:
            logger.info(f"    ✓ git apply --check PASSED")
            final_patch = patch_text
            final_pairs = pairs
            final_warnings = warnings
            break
        
        # FIX: 空 patch 或 apply fail 都触发 retry (而非 silently break)
        if attempt_idx < max_retries:
            logger.warning(f"    ✗ {apply_error}, retrying...")
            # 构造 retry prompt 反馈具体失败信息
            retry_feedback = f"""

PREVIOUS ATTEMPT FAILED.

Failure type: {('empty patch (no anchored pairs)' if not patch_text else 'git apply rejected')}
Details: {apply_error}

What went wrong (likely causes):
"""
            if not patch_text:
                retry_feedback += (
                    "- Your BEFORE blocks did not match any code in the retrieved sources.\n"
                    "- Make sure to COPY BEFORE byte-for-byte from the 'Code Context' section above.\n"
                    "- Include leading whitespace / indentation.\n"
                    "- Do not paraphrase or summarize.\n"
                )
            else:
                retry_feedback += (
                    "- git apply rejected the synthesized patch.\n"
                    "- Likely your AFTER block has structural issues.\n"
                    "- Make sure AFTER is valid Python with consistent indentation.\n"
                )
            retry_feedback += "\nTry again, paying close attention to byte-exact BEFORE matching.\n\n## Your Output:\n"
            current_prompt = prompt + retry_feedback
            # 保留 final_patch 用 last attempt (空) 也算输出
            final_patch = patch_text
            final_pairs = pairs
            final_warnings = warnings
            continue
        else:
            # 最后一次, 接受当前结果
            logger.warning(f"    ✗ Final attempt failed: {apply_error}")
            final_patch = patch_text
            final_pairs = pairs
            final_warnings = warnings
            break
    
    # Save final patch
    patch_path = task_dir / "generated_patch.diff"
    patch_path.write_text(final_patch if final_patch else "")
    
    # Save metadata
    metadata = {
        'task_id': task['instance_id'],
        'model': model_name,
        'top_k': top_k,
        'max_retries': max_retries,
        'attempts': attempt_log,
        'final_pairs': [
            {
                'raw_before': (p.raw_before or '')[:200],
                'match_status': p.match_status,
                'matched_file': p.matched_file,
                'matched_start_line': p.matched_start_line,
            }
            for p in final_pairs
        ],
        'final_warnings': final_warnings,
        'final_patch_size': len(final_patch),
        'final_git_apply_ok': any(a['git_apply_ok'] for a in attempt_log),
    }
    metadata_path = task_dir / "anchor_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    
    return metadata


# ============================================================================
# Reconstruct nodes from KTF (for anchor_diff)
# ============================================================================

def get_retrieved_node_objects(retrieved_nodes: list, ktf_path: Path) -> list:
    """retrieved 是 KnowledgeNode list, 已经含 source_code, 直接返回.
    
    (此函数留作 future: 如果 retrieved.json 仅含 inject 不含 node 对象, 在此 reconstruct)
    """
    return retrieved_nodes


# ============================================================================
# Main entry
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Task selection (mutually exclusive)
    sel_group = parser.add_argument_group("Task selection (choose one mode)")
    sel_group.add_argument("--tasks", nargs="+",
                            help="指定 instance_ids, e.g. django__django-11001")
    sel_group.add_argument("--candidates",
                            help="从已有 candidates JSON 读 (e.g. day5_candidates.json)")
    sel_group.add_argument("--select-easy", type=int, default=0,
                            help="从 SWE-bench Lite 选 N 道 easy")
    sel_group.add_argument("--select-medium", type=int, default=0,
                            help="从 SWE-bench Lite 选 N 道 medium")
    sel_group.add_argument("--select-hard", type=int, default=0,
                            help="从 SWE-bench Lite 选 N 道 hard")
    sel_group.add_argument("--filter-repos", nargs="+",
                            help="只选这些 repos (e.g. astropy/astropy django/django)")
    
    # Model
    parser.add_argument("--model", choices=['r1', 'nemotron'], default='r1',
                        help="generator 模型 (default: r1)")
    parser.add_argument("--r1-lora-path", default="models/explorer-grpo-sanity/checkpoint-50",
                        help="R1 LoRA 路径")
    parser.add_argument("--nemotron-path", default="./models/nemotron-nano-9b-v2",
                        help="Nemotron 模型路径")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    
    # Pipeline params
    parser.add_argument("--top-k", type=int, default=3,
                        help="retrieve top-k (default 3)")
    parser.add_argument("--retriever", choices=['bm25', 'graph_expanded'], default='bm25',
                        help="检索器: bm25 (baseline) | graph_expanded (Day 8 召回扩展)")
    parser.add_argument("--seed-k", type=int, default=3,
                        help="graph_expanded 的 BM25 seed 数 (default 3)")
    parser.add_argument("--max-expansion", type=int, default=20,
                        help="graph_expanded 的最大扩展邻居数 (default 20)")
    parser.add_argument("--localize", action="store_true",
                        help="两阶段: 先召回 candidate-k 候选, LLM 选 top-k, 再生成 (Day 9)")
    parser.add_argument("--candidate-k", type=int, default=15,
                        help="localize 阶段召回的候选数 (default 15, 然后 LLM 选 top-k)")
    parser.add_argument("--retry", type=int, default=2,
                        help="generation retry 次数 (default 2)")
    parser.add_argument("--work-dir", default="/tmp/swe-bench-day7",
                        help="工作目录 (含 per-task 子目录)")
    parser.add_argument("--skip-build", action="store_true",
                        help="跳过 build KTF (假设已存在)")
    parser.add_argument("--skip-clone", action="store_true",
                        help="跳过 clone repo (假设已存在)")
    parser.add_argument("--candidates-output", default="day7_candidates.json",
                        help="保存 selected tasks 到这个文件")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    # === 1. Select tasks ===
    if args.tasks:
        logger.info(f"[1] Loading {len(args.tasks)} tasks by instance_id")
        raw_tasks = filter_tasks_by_ids(args.tasks)
    elif args.candidates:
        logger.info(f"[1] Loading from candidates file: {args.candidates}")
        cands_data = json.loads(Path(args.candidates).read_text())
        raw_tasks = []
        for k, v in cands_data.items():
            if v is not None and 'instance_id' in v:
                raw_tasks.append(v)
    elif args.select_easy or args.select_medium or args.select_hard:
        logger.info(f"[1] Selecting from SWE-bench Lite: "
                    f"easy={args.select_easy}, medium={args.select_medium}, hard={args.select_hard}")
        raw_tasks = select_tasks(
            n_easy=args.select_easy,
            n_medium=args.select_medium,
            n_hard=args.select_hard,
            repos=args.filter_repos,
        )
    else:
        print("❌ Must specify one of: --tasks, --candidates, --select-{easy|medium|hard}")
        return 1
    
    if not raw_tasks:
        print("❌ No tasks selected")
        return 1
    
    tasks = [task_to_record(t) if 'difficulty' not in t else t for t in raw_tasks]
    
    # Save selected tasks to candidates file
    cands_dict = {
        f"task_{i}": t for i, t in enumerate(tasks)
    }
    Path(args.candidates_output).write_text(
        json.dumps(cands_dict, indent=2, ensure_ascii=False)
    )
    logger.info(f"  Saved {len(tasks)} tasks to {args.candidates_output}")
    
    print()
    print("=" * 70)
    print(f"Day 7 Pipeline: {len(tasks)} tasks, model={args.model}")
    print("=" * 70)
    for t in tasks:
        diff = t.get('difficulty', '?')
        print(f"  [{diff}] {t['instance_id']} ({t['repo']})")
    print()
    
    # === 2. Load model (once for all tasks) ===
    if args.model == 'r1':
        from knowledge_tree.local_model_clients import make_r1_generator
        logger.info(f"[2] Loading R1-Distill-14B + LoRA from {args.r1_lora_path}")
        model_callable = make_r1_generator(
            lora_path=args.r1_lora_path,
            max_new_tokens=args.max_new_tokens,
            verbose=args.verbose,
        )
    else:
        from knowledge_tree.local_model_clients import make_nemotron_retriever
        logger.info(f"[2] Loading Nemotron-Nano-9B-v2 from {args.nemotron_path}")
        # Nemotron 作为 generator 用 (虽然函数名是 retriever, 但只是参数预设)
        from knowledge_tree.local_model_clients import LocalModelCallable
        model_callable = LocalModelCallable(
            base_model=args.nemotron_path,
            use_int4=True,
            max_new_tokens=args.max_new_tokens,
            temperature=0.6,  # generator 温度
            keep_thinking=False,
            verbose=args.verbose,
        )
    
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # === 3. Per-task pipeline ===
    summary = []
    for i, task in enumerate(tasks):
        task_id = task['instance_id']
        task_dir = work_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        
        print()
        print("─" * 70)
        print(f"[{i+1}/{len(tasks)}] {task_id} ({task.get('difficulty', '?')})")
        print("─" * 70)
        
        task_result = {
            'task_id': task_id,
            'repo': task['repo'],
            'difficulty': task.get('difficulty', 'unknown'),
            'status': 'pending',
            'errors': [],
        }
        
        try:
            # Step 1: Clone
            if args.skip_clone:
                repo_path = task_dir / "repo"
                if not repo_path.exists():
                    raise RuntimeError(f"--skip-clone but {repo_path} not found")
                logger.info(f"  [step 1] skip clone (--skip-clone)")
            else:
                logger.info(f"  [step 1] Clone + checkout")
                repo_path = step_clone_repo(task, task_dir)
            
            # Step 2: Build KTF
            ktf_path = task_dir / "ktf.json"
            logger.info(f"  [step 2] Build KTF")
            n_nodes = step_build_ktf(task, repo_path, ktf_path, skip_if_exists=args.skip_build)
            task_result['ktf_nodes'] = n_nodes
            
            # Step 3: retrieve (若 localize 开启, 召回 candidate_k 个候选)
            n_retrieve = args.candidate_k if args.localize else args.top_k
            logger.info(f"  [step 3] retrieve ({args.retriever}) "
                        f"{'candidate-k=' + str(n_retrieve) + ' (localize)' if args.localize else 'top-' + str(args.top_k)}")
            retrieved_path = task_dir / "retrieved.json"
            from knowledge_tree.storage import JSONStorage
            from knowledge_tree.core import KnowledgeTree
            from knowledge_tree.retrievers import BM25Retriever, GraphExpandedRetriever
            storage = JSONStorage(str(ktf_path), create_if_missing=False)
            tree = KnowledgeTree(storage.list_all())
            if args.retriever == 'graph_expanded':
                retriever = GraphExpandedRetriever(
                    tree, seed_k=args.seed_k, max_expansion=args.max_expansion,
                )
                retrieved_nodes = retriever.retrieve(task['problem_statement'], top_k=n_retrieve)
                try:
                    prov = retriever.retrieve_with_provenance(task['problem_statement'], top_k=n_retrieve)
                    (task_dir / "retrieve_provenance.json").write_text(json.dumps([
                        {'qualified_name': p['qualified_name'], 'provenance': p['provenance']}
                        for p in prov
                    ], indent=2, ensure_ascii=False))
                except Exception as e:
                    logger.warning(f"  provenance dump failed: {e}")
            else:
                bm25 = BM25Retriever(tree)
                retrieved_nodes = bm25.retrieve(task['problem_statement'], top_k=n_retrieve)

            # Step 3.5: 两阶段定位 (Stage 1: LLM 从 candidate 选相关 function)
            if args.localize and len(retrieved_nodes) > args.top_k:
                from knowledge_tree.localizer import localize, reorder_by_localization
                logger.info(f"  [step 3.5] localize: LLM 从 {len(retrieved_nodes)} 候选选 {args.top_k} 个")
                loc_result = localize(
                    task['problem_statement'], retrieved_nodes,
                    model_callable, select_k=args.top_k,
                )
                logger.info(f"    selected: {loc_result.selected_ids}")
                logger.info(f"    reasoning: {loc_result.reasoning[:120]}")
                if loc_result.fell_back:
                    logger.warning(f"    ⚠ localization fell back to top-{args.top_k}")
                # 重排: 选中的在前, 其余补后 (anchor fallback 仍可用)
                retrieved_nodes = reorder_by_localization(retrieved_nodes, loc_result)
                # 保存 localization 诊断
                (task_dir / "localization.json").write_text(json.dumps({
                    'selected_ids': loc_result.selected_ids,
                    'reasoning': loc_result.reasoning,
                    'n_candidates': loc_result.n_candidates,
                    'fell_back': loc_result.fell_back,
                }, indent=2, ensure_ascii=False))
                # Stage 2 只用前 top_k 个 (选中的) 喂 generation
                retrieved_nodes = retrieved_nodes[:args.top_k]
            
            # 写 retrieved.json (兼容性)
            retrieved_data = [{
                'rank': j + 1,
                'id': n.id,
                'inject': n.llm_inject_text(),
            } for j, n in enumerate(retrieved_nodes)]
            retrieved_path.write_text(json.dumps(retrieved_data, indent=2, ensure_ascii=False))
            
            # Oracle hit
            oracle_files = re.findall(r'diff --git a/(\S+)', task['patch'])
            hits = sum(
                1 for n in retrieved_nodes
                if any(of in n.domain_metadata.get('file', '') for of in oracle_files)
            )
            task_result['oracle_files'] = oracle_files
            task_result['oracle_hits'] = hits
            logger.info(f"    Retrieved: {[n.id for n in retrieved_nodes]}")
            logger.info(f"    Oracle hits: {hits}/{args.top_k}")
            
            # Step 5: Anchor-based generation + synth
            logger.info(f"  [step 5] Anchor-based generation ({args.model})")
            gen_metadata = step_generate_and_synth(
                task, retrieved_nodes, repo_path, task_dir,
                model_callable, args.model, args.top_k, args.retry,
            )
            task_result['final_patch_size'] = gen_metadata['final_patch_size']
            task_result['final_git_apply_ok'] = gen_metadata['final_git_apply_ok']
            task_result['n_attempts'] = len(gen_metadata['attempts'])
            task_result['final_pair_statuses'] = [
                p['match_status'] for p in gen_metadata['final_pairs']
            ]
            task_result['status'] = 'success' if gen_metadata['final_git_apply_ok'] else 'apply_failed'
            
            print(f"  ✓ Done: {task_result['status']}, "
                  f"patch={gen_metadata['final_patch_size']} chars, "
                  f"apply={gen_metadata['final_git_apply_ok']}, "
                  f"attempts={len(gen_metadata['attempts'])}")
        
        except Exception as e:
            logger.exception(f"Task failed: {task_id}")
            task_result['status'] = 'error'
            task_result['errors'].append(str(e))
            print(f"  ✗ Error: {e}")
        
        summary.append(task_result)
        
        # Per-task progress dump (中途 fail 也保留)
        (work_dir / "day7_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False)
        )
    
    # === 4. Unload model ===
    if hasattr(model_callable, 'unload'):
        model_callable.unload()
    
    # === 5. Summary ===
    print()
    print("=" * 70)
    print("Pipeline Summary")
    print("=" * 70)
    
    by_status = defaultdict(int)
    by_difficulty = defaultdict(lambda: defaultdict(int))
    for r in summary:
        by_status[r['status']] += 1
        by_difficulty[r['difficulty']][r['status']] += 1
    
    print(f"\nTotal: {len(summary)} tasks")
    print(f"  Apply OK: {by_status.get('success', 0)}")
    print(f"  Apply failed: {by_status.get('apply_failed', 0)}")
    print(f"  Error: {by_status.get('error', 0)}")
    
    print(f"\nBy difficulty:")
    for diff in ['easy', 'medium', 'hard', 'unknown']:
        stats = by_difficulty.get(diff, {})
        total = sum(stats.values())
        if total == 0:
            continue
        ok = stats.get('success', 0)
        print(f"  {diff}: {ok}/{total} apply OK")
    
    print(f"\nRetrieval (oracle file hits):")
    for r in summary:
        if r['status'] == 'error':
            continue
        oracle_hits = r.get('oracle_hits', 0)
        top_k = args.top_k
        print(f"  {r['task_id']}: {oracle_hits}/{top_k} oracle file hits, "
              f"pair_statuses={r.get('final_pair_statuses', [])}")
    
    print(f"\nWork dir: {work_dir}")
    print(f"Summary: {work_dir}/day7_summary.json")
    print(f"Candidates: {args.candidates_output}")
    
    print()
    print("Next steps:")
    print("  1. Analyze pair_statuses 分布: exact vs indent_corrected vs fuzzy_line_anchor vs not_found")
    print("  2. Analyze oracle_hits vs apply_ok 相关性 (是否 retrieve 不命中也能 apply ok)")
    print("  3. 准备 predictions.jsonl 跑 swebench harness:")
    print(f"     python scripts/day5_step6_swebench_eval.py --prepare --patches-dir {work_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
