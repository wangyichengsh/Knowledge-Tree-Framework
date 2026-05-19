#!/usr/bin/env python3
"""
scripts/fix_children_ids.py
============================

Phase 4.1 Week 2 后处理 - 修复 topic.children_ids 字段.

背景 (PROTO-7.4 实测发现):
  Phase 4.1 Week 2 跑剩余节点时, build_tree_with_hierarchy v1 bug:
    第二次 build 时, parent.children_ids 被覆盖而非合并
    导致 algebra/geometry 等 topic 的 children_ids 只含本次新增的 1 个,
    而非全部 children
  
  Tree.json 中体现:
    algebra.children_ids = [nested_radicals_simplification]  # 应该 20 个
    导致 validate 报 19 个"双向不一致"

修复 (本脚本):
  扫描所有节点的 parent_id, 反向重建每个 topic 的 children_ids
  写回 storage
  
  这是数据修复, 不调 LLM, 免费, 几秒.

PROTO 关联:
  PROTO-7.4 (实测校准): 真实 tree.json 实测发现的 bug
  PROTO-7.9 (单测 + 实数据 dual validation): 修复前后 validate 应通过

用法:
  python scripts/fix_children_ids.py \\
    --tree-json knowledge_tree/docs/math/tree.json \\
    --output knowledge_tree/docs/math/tree.json

  # 干跑 (不写回)
  python scripts/fix_children_ids.py \\
    --tree-json knowledge_tree/docs/math/tree.json \\
    --dry-run
"""

import argparse
import logging
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_tree.core import KnowledgeTree
from knowledge_tree.storage import JSONStorage

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tree-json", required=True)
    parser.add_argument("--output", default=None,
                        help="默认 in-place 写回 --tree-json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 78)
    print("Fix Children IDs - 修复 topic.children_ids 双向关系")
    print("=" * 78)

    storage = JSONStorage(args.tree_json, create_if_missing=False)
    nodes = storage.list_all()
    print(f"\n加载 {len(nodes)} 节点")

    # === 反向扫描: parent_id -> children ids ===
    parent_to_children: dict[str, list[str]] = defaultdict(list)
    nodes_by_id = {n.id: n for n in nodes}

    for node in nodes:
        if node.parent_id:
            if node.parent_id not in nodes_by_id:
                logger.warning(
                    "节点 %r 的 parent_id=%r 不存在于 tree (orphan)",
                    node.id, node.parent_id,
                )
                continue
            parent_to_children[node.parent_id].append(node.id)

    # === Pre-fix 状态 ===
    print(f"\n=== Pre-fix 状态 ===")
    pre_validate = KnowledgeTree.from_storage(storage).validate(strict=False)
    print(f"  双向不一致问题: {len(pre_validate)}")

    print(f"\n  各 topic 当前 children_ids 数量:")
    for parent_id in sorted(parent_to_children.keys()):
        parent_node = nodes_by_id[parent_id]
        actual = len(parent_node.children_ids or [])
        should = len(parent_to_children[parent_id])
        marker = "✓" if actual == should else "✗"
        print(f"    {marker} {parent_id}: actual={actual}, should={should}")

    # === 修复 ===
    print(f"\n=== Fix 阶段 ===")
    fixed_count = 0
    for parent_id, children_ids in parent_to_children.items():
        parent_node = nodes_by_id[parent_id]
        old_children = list(parent_node.children_ids or [])
        # 合并 (保留可能有的 extra, 加上扫描出来的) 然后按 id 排序去重
        new_children = sorted(set(old_children) | set(children_ids))
        if new_children != old_children:
            fixed_count += 1
            print(f"  {parent_id}: {len(old_children)} → {len(new_children)} children")
            parent_node.children_ids = new_children
            if not args.dry_run:
                storage.save_node(parent_node)

    if not args.dry_run:
        storage.flush()

    # === Post-fix 验证 ===
    print(f"\n=== Post-fix 验证 ===")
    if args.dry_run:
        print(f"  [Dry-run] 未写回, 跳过 validate")
    else:
        # 重新加载 (确认 flush 生效)
        storage2 = JSONStorage(args.tree_json, create_if_missing=False)
        tree2 = KnowledgeTree.from_storage(storage2)
        post_validate = tree2.validate(strict=False)
        print(f"  双向不一致问题: {len(post_validate)}")
        if post_validate:
            print(f"  ⚠️ 仍有问题:")
            for iss in post_validate[:5]:
                print(f"    - {iss}")
        else:
            print(f"  ✅ Validate 完全通过")

    # === 总结 ===
    print(f"\n=== 总结 ===")
    print(f"  修复 {fixed_count} 个 topic 节点的 children_ids")
    if args.dry_run:
        print(f"  [Dry-run] 未写回. 用 --output 或去掉 --dry-run 启用写回")
    else:
        output_path = args.output or args.tree_json
        print(f"  写回: {output_path}")


if __name__ == "__main__":
    main()
