#!/usr/bin/env python3
"""
scripts/build_polars_auto_tree.py
====================================

Phase 4.2 Stage 1.3b: 用 PythonAPIBuilder 从 Polars 安装包**自动** build KTF.

vs build_polars_mini_tree.py (Stage 1.2 手工):
  - 手工: 32 节点, ~3h 工作量, 高质量节点 + 手写 pitfalls
  - 自动: 35 节点, ~5s 工作量, 完整 docstring + auto-detected pitfalls

实验目的:
  1. 验证 PythonAPIBuilder 可以工程化构造 KTF
  2. 对照手工 vs 自动 KTF 的 retrieval quality + B_hybrid accuracy
  3. 为 Phase 4.3 SWE-bench (AST-based) 铺路

用法:
  python scripts/build_polars_auto_tree.py \\
    --output knowledge_tree/docs/polars/tree_auto.json

  # 与手工对比 (50 题, R1)
  python scripts/run_polars_pilot.py \\
    --tree-json knowledge_tree/docs/polars/tree_auto.json \\
    --output polars_50_r1_auto_tree.jsonl \\
    --base-model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \\
    --explorer-lora models/explorer-grpo-sanity/checkpoint-50 \\
    --use-int4 \\
    --api-key $ANTHROPIC_API_KEY

PROTO 关联:
  PROTO-7.1 (grep 复用): 复用 PythonAPIBuilder + KnowledgeStorage
  PROTO-7.4 (实测校准): 对照手工 32 节点 KTF (BM25 命中, B-F 真效用)
  PROTO-7.20 (grep outputs): JSON 输出与手工 tree 同格式, 可直接互换
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 50 题需要的 API 清单 (与手工 32 节点对应, 可能稍多/少)
POLARS_API_LIST = [
    # ====== Lazy I/O (6) ======
    "pl.scan_csv",
    "pl.scan_parquet",
    "pl.scan_ndjson",
    "LazyFrame.sink_parquet",
    "LazyFrame.sink_csv",
    "LazyFrame.collect",
    
    # ====== Expression core (11) ======
    "pl.col",
    "pl.when",
    "pl.lit",
    "Expr.alias",
    "Expr.cast",
    "Expr.fill_null",
    "Expr.is_null",
    "Expr.is_not_null",
    "Expr.diff",
    "Expr.cum_sum",
    "Expr.rank",
    
    # ====== DataFrame methods (10) ======
    "DataFrame.with_columns",
    "DataFrame.filter",
    "DataFrame.select",
    "DataFrame.group_by",
    "DataFrame.group_by_dynamic",
    "DataFrame.rolling",
    "DataFrame.join",
    "DataFrame.join_asof",
    "DataFrame.sort",
    "DataFrame.lazy",
    
    # ====== Reshape (4) ======
    "DataFrame.pivot",
    "DataFrame.unpivot",
    "DataFrame.explode",
    "pl.concat",
    
    # ====== Namespaces (6) ======
    "Expr.str.contains",
    "Expr.str.split",
    "Expr.str.len_chars",
    "Expr.dt.year",
    "Expr.dt.month",
    "Expr.dt.truncate",
    "Expr.dt.offset_by",
    "Expr.list.eval",
    "Expr.list.len",
    "Expr.struct.field",
    
    # ====== Lazy ops (2) ======
    "LazyFrame.explain",
    "LazyFrame.filter",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True,
                        help="Output JSON path (e.g. tree_auto.json)")
    parser.add_argument("--summary-only", action="store_true",
                        help="不写文件, 仅打印 summary")
    parser.add_argument("--enrich-with-llm", action="store_true",
                        help="用 LLM 补充 pitfalls / examples (需 ANTHROPIC_API_KEY 或本地模型)")
    parser.add_argument("--api-key", default=None,
                        help="Anthropic API key (用于 --enrich-with-llm)")
    args = parser.parse_args()

    from knowledge_tree.python_api_builder import PythonAPIBuilder

    # Optional LLM enrichment
    llm = None
    if args.enrich_with_llm:
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            from knowledge_tree.llm_clients import ClaudeCallable
            llm = ClaudeCallable(api_key=api_key, max_tokens=1024, verbose=False)
            print(f"Using Claude API for LLM enrichment")
        else:
            print("[Warning] --enrich-with-llm but no API key, skipping enrichment")

    print("=" * 70)
    print(f"Stage 1.3b: PythonAPIBuilder build (auto KTF)")
    print("=" * 70)
    print(f"  API list: {len(POLARS_API_LIST)} APIs")
    print(f"  LLM enrichment: {llm is not None}")

    builder = PythonAPIBuilder(
        llm_callable=llm,
        enrich_with_llm=(llm is not None),
    )

    import time
    t_start = time.time()
    nodes = builder.build_from_api_list(POLARS_API_LIST, skip_on_failure=True)
    t_elapsed = time.time() - t_start

    print(f"\n构建完成 in {t_elapsed:.1f}s:")
    print(f"  成功: {builder.total_built}/{len(POLARS_API_LIST)}")
    print(f"  失败: {builder.total_failed}")
    print(f"  LLM calls: {builder.total_llm_calls}")

    if builder.total_failed > 0:
        print(f"\n  Failed APIs (检查 polars version 兼容):")
        attempted = set(POLARS_API_LIST)
        built = set(n.domain_metadata.get('api_name') for n in nodes)
        for failed in (attempted - built):
            print(f"    - {failed}")

    # Stats
    if nodes:
        total_chars = sum(len(n.llm_inject_text()) for n in nodes)
        print(f"\n节点统计:")
        print(f"  Avg inject chars: {total_chars / len(nodes):.0f}")
        print(f"  Total inject: {total_chars} chars (~{total_chars // 4} tokens)")
        
        # Pitfalls coverage
        with_pitfalls = sum(1 for n in nodes if n.common_pitfalls)
        print(f"  Nodes with auto pitfalls: {with_pitfalls}/{len(nodes)}")

    # Save
    if not args.summary_only:
        from knowledge_tree.storage import JSONStorage
        # 删除已有文件
        if os.path.exists(args.output):
            os.remove(args.output)
        # autosave=True: 每次 save_node 后立即 flush
        storage = JSONStorage(args.output, create_if_missing=True, autosave=False)
        for n in nodes:
            storage.save_node(n)
        storage.flush()  # 显式 flush
        print(f"\n  Saved to: {args.output}")
        
        # Validate by re-loading
        from knowledge_tree.core import KnowledgeTree
        reload = JSONStorage(args.output, create_if_missing=False)
        tree = KnowledgeTree(reload.list_all())
        print(f"  Tree validated: {len(tree)} nodes loaded")


if __name__ == "__main__":
    main()
