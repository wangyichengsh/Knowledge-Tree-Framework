#!/usr/bin/env python3
"""
scripts/day12_enrich_summary.py
================================

Phase 4.3 Day 12: KTF summary 富化 — 攻"词汇鸿沟"残差.

动机 (实测诊断, framework T-3.19 关联):
  KTF 节点的 BM25 索引内容 (definition/key_facts) 几乎全是结构元数据
  (函数名/路径/行号/签名), 没有"函数做什么"的自然语言. 而 problem_statement
  用行为描述 bug. 词汇空间不重叠 → miss 题 (rank 100-2800) 和 file_only
  oracle rank 远的根因.

  本脚本用 LLM 把代码"翻译"成客观行为描述, 回填 domain_metadata['llm_summary'],
  让 BM25 能匹配 problem 的行为词汇.

关键设计 (基于批判性评估):
  1. summary 与问题【解耦】: 只描述函数客观行为, 绝不针对任何具体 problem
     (否则是过拟合/泄漏). prompt 强制这一点.
  2. method/function → llm_summary (第一轮 A/B 用, 攻"行为型"残差)
  3. class → class_summary (生成存着, 第二轮救援用, 攻"结构型解耦题")
     class_summary 描述: 该类的职责 + 定义了哪些方法 + 代码组织模式
  4. 不覆写 definition (保留结构信息, 可回退); llm_summary 是独立字段
  5. 翻译任务非推理任务 → 本地模型/Sonnet/Haiku 即可, 不必 Opus

用法:
  # 富化单个 KTF (method + class 都生成)
  python scripts/day12_enrich_summary.py \\
      --ktf /tmp/swe-bench-day10/astropy__astropy-7746/ktf.json \\
      --model claude_api --claude-model claude-sonnet-4-6

  # 批量富化 (work-dir 下所有 ktf.json)
  python scripts/day12_enrich_summary.py \\
      --work-dir /tmp/swe-bench-day10 --model claude_api \\
      --claude-model claude-sonnet-4-6

  富化后用 sweep A/B 对比:
  python scripts/day10_retrieval_sweep.py --candidates day10_balanced50.json \\
      --work-dir /tmp/swe-bench-day10 --skip-build  # baseline
  python scripts/day10_retrieval_sweep.py ... --include-llm-summary  # enriched
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get('PROJECT_ROOT', '.'))
logger = logging.getLogger(__name__)


METHOD_SUMMARY_PROMPT = """You are documenting a Python function for a code search index.

Write a 1-2 sentence OBJECTIVE description of what this function DOES — its behavior,
purpose, inputs/outputs, and side effects. Use natural language a developer would use
when describing the BEHAVIOR (e.g. "removes figures from the registry", "converts
pixel coordinates to world coordinates", "validates that input categories match").

CRITICAL RULES:
- Describe ONLY what the code objectively does. Do NOT speculate about bugs or fixes.
- Use behavioral verbs and domain nouns that someone REPORTING A BUG would naturally use.
- Do NOT just restate the function name. Explain the actual behavior.
- No preamble, no markdown. Just the description sentence(s).

Function: {qualified_name}
File: {file}

```python
{source_code}
```

Behavior description:"""


CLASS_SUMMARY_PROMPT = """You are documenting a Python class for a code search index.

Write a 2-3 sentence OBJECTIVE description of this class covering:
- Its responsibility / what it represents
- The KEY methods it defines (list the important method names)
- Its code organization pattern if notable (e.g. "defines magic methods via a proxy
  helper", "implements the encoder interface with fit/transform methods")

CRITICAL RULES:
- Describe ONLY what the class objectively is/does. No bug/fix speculation.
- Mention method names a developer might look for when needing to ADD or MODIFY behavior.
- No preamble, no markdown. Just the description.

Class: {qualified_name}
File: {file}

```python
{source_code}
```

Class description:"""


def enrich_ktf(ktf_path: Path, model_callable, only_empty: bool = True,
               do_class: bool = True, max_source_chars: int = 4000,
               verbose: bool = False) -> dict:
    """富化单个 KTF 的 llm_summary / class_summary. 原地写回.

    Returns: 统计 dict.
    """
    from knowledge_tree.storage import JSONStorage

    storage = JSONStorage(str(ktf_path), create_if_missing=False, autosave=False)
    nodes = storage.list_all()

    stats = {'total': len(nodes), 'method_enriched': 0, 'class_enriched': 0,
             'skipped_existing': 0, 'errors': 0}

    for node in nodes:
        dm = node.domain_metadata or {}
        ntype = dm.get('type', '')
        src = (node.source_code or '')[:max_source_chars]
        if not src.strip():
            continue

        try:
            if ntype in ('method', 'function'):
                if only_empty and dm.get('llm_summary'):
                    stats['skipped_existing'] += 1
                    continue
                prompt = METHOD_SUMMARY_PROMPT.format(
                    qualified_name=dm.get('qualified_name', node.title),
                    file=dm.get('file', '?'), source_code=src)
                summary = model_callable(prompt).strip()
                dm['llm_summary'] = summary
                node.domain_metadata = dm
                storage.save_node(node)
                stats['method_enriched'] += 1
                if verbose:
                    logger.info("  [%s] %s", dm.get('qualified_name'), summary[:80])

            elif ntype == 'class' and do_class:
                if only_empty and dm.get('class_summary'):
                    stats['skipped_existing'] += 1
                    continue
                prompt = CLASS_SUMMARY_PROMPT.format(
                    qualified_name=dm.get('qualified_name', node.title),
                    file=dm.get('file', '?'), source_code=src)
                summary = model_callable(prompt).strip()
                dm['class_summary'] = summary
                node.domain_metadata = dm
                storage.save_node(node)
                stats['class_enriched'] += 1
        except Exception as e:
            logger.warning("enrich failed for %s: %s", node.id, e)
            stats['errors'] += 1

    storage.flush()
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ktf", help="单个 ktf.json 路径")
    parser.add_argument("--work-dir", help="批量: work-dir 下所有 <task>/ktf.json")
    parser.add_argument("--model", default="claude_api",
                        choices=['r1', 'nemotron', 'claude_api'])
    parser.add_argument("--claude-model", default="claude-sonnet-4-6",
                        help="翻译任务, sonnet/haiku 即可, 不必 opus")
    parser.add_argument("--no-class", action="store_true",
                        help="不生成 class_summary (只 method/function)")
    parser.add_argument("--re-enrich", action="store_true",
                        help="覆写已有 summary (默认只填空)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # 加载模型
    if args.model == 'claude_api':
        from knowledge_tree.claude_api_client import ClaudeAPICallable
        model_callable = ClaudeAPICallable(model=args.claude_model,
                                            max_tokens=512,  # summary 短
                                            verbose=args.verbose)
    elif args.model == 'r1':
        from knowledge_tree.local_model_clients import make_r1_generator
        model_callable = make_r1_generator(max_new_tokens=256, verbose=args.verbose)
    else:
        from knowledge_tree.local_model_clients import LocalModelCallable
        model_callable = LocalModelCallable(base_model="./models/nemotron-nano-9b-v2",
                                            use_int4=True, max_new_tokens=256,
                                            verbose=args.verbose)

    # 收集 KTF 路径
    ktf_paths = []
    if args.ktf:
        ktf_paths = [Path(args.ktf)]
    elif args.work_dir:
        ktf_paths = sorted(Path(args.work_dir).glob("*/ktf.json"))
    else:
        print("需要 --ktf 或 --work-dir")
        return 1

    print("=" * 70)
    print(f"Day 12: Summary 富化 ({len(ktf_paths)} 个 KTF, model={args.claude_model})")
    print("=" * 70)

    total_stats = {'method_enriched': 0, 'class_enriched': 0, 'errors': 0}
    for i, kp in enumerate(ktf_paths):
        print(f"\n[{i+1}/{len(ktf_paths)}] {kp.parent.name}")
        stats = enrich_ktf(kp, model_callable, only_empty=not args.re_enrich,
                           do_class=not args.no_class, verbose=args.verbose)
        print(f"  method+func: {stats['method_enriched']}, class: {stats['class_enriched']}, "
              f"skip: {stats['skipped_existing']}, err: {stats['errors']}")
        for k in total_stats:
            total_stats[k] += stats.get(k, 0)

    if hasattr(model_callable, 'unload'):
        model_callable.unload()

    print(f"\n{'='*70}")
    print(f"富化完成: method+func {total_stats['method_enriched']}, "
          f"class {total_stats['class_enriched']}, errors {total_stats['errors']}")
    print(f"\n下一步: A/B sweep 对比")
    print(f"  baseline:  python scripts/day10_retrieval_sweep.py --candidates <c> --work-dir <w> --skip-build")
    print(f"  enriched:  同上 + --include-llm-summary")
    return 0


if __name__ == "__main__":
    sys.exit(main())
