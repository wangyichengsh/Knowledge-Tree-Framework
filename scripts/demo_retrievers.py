"""
scripts/demo_retrievers.py
==========================

6 conditions retrievers 实数据演示 (PROTO-7.9 dual validation 第二半).

目的:
  (1) 单元测试通过 ≠ 实数据 work, 必须实际跑一遍 6 conditions
  (2) 用 demo_basic.py 同样的 domino path 节点 (3 个)
  (3) 模拟一个数学题 query, 看 6 retrievers 各返回什么
  (4) 用本地 deterministic mock LLM (返回固定 IDs), 验证 LLM-dep retrievers 工作

为什么不调真 Claude API:
  - 演示目的: 验证代码 plumbing 正确, 不验证 LLM 决策质量
  - LLM 决策质量 = Phase 4.1 Week 3 真实实验测的
  - 本地 mock = deterministic, 可重现

运行:
  cd /home/claude && python scripts/demo_retrievers.py
"""

import logging
import os
import sys

# 路径设置
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 复用 demo_basic 的节点构造
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo_basic import make_domino_path_nodes

from knowledge_tree.core import KnowledgeTree
from knowledge_tree.retrievers import make_all_retrievers


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


class DemoLLM:
    """
    Demo 用 mock LLM. 简单策略:
      - 看 prompt 含哪个节点 id, 按出现顺序返回前 3 个

    这是 deterministic / 可重现的, 用于验证 plumbing.
    真实 Phase 4.1 Week 3 实验中, 此处替换为 Claude API callable.
    """

    def __init__(self, tree: KnowledgeTree) -> None:
        self.tree = tree
        self.call_count = 0
        # 所有节点 ids (用于检测哪些在 prompt 中)
        self.all_ids = [n.id for n in tree.list_all()]

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        # 简单策略: 找 prompt 中出现的 node ids, 返回前 3
        # 注意: 因为 prompt 含完整 listing, 所有节点都会出现; 取前 3
        appearing = []
        for nid in self.all_ids:
            if nid in prompt:
                # 找出位置, 用第一次出现位置排序 (deterministic)
                idx = prompt.find(nid)
                appearing.append((nid, idx))
        # 按出现位置排序
        appearing.sort(key=lambda x: x[1])
        # 模拟 "LLM 决策": 选 prompt 中前 3 个出现的 id
        selected = [nid for nid, _ in appearing[:3]]
        return '{"selected_ids": ' + str(selected).replace("'", '"') + "}"


def demo() -> None:
    print("=" * 78)
    print("KTF v2 - 6 conditions retrievers 实数据演示")
    print("=" * 78)

    # === Setup ===
    nodes = make_domino_path_nodes()
    tree = KnowledgeTree(nodes=nodes)
    print(f"\n[Setup] 构造 KnowledgeTree: {len(tree)} 节点")
    print(f"  节点 ids: {sorted(n.id for n in tree.list_all())}")

    # 一个模拟 query (近似 domino path 测试题)
    query = (
        "How many ways to tile a 4x3 grid with horizontal dominoes "
        "such that they form a monotone path?"
    )
    print(f"\n[Query]: {query}")

    # 构造 retrievers
    llm = DemoLLM(tree)
    retrievers = make_all_retrievers(tree, llm)

    # === 跑 6 conditions ===
    print(f"\n{'='*78}")
    print(f"6 Conditions 检索结果:")
    print(f"{'='*78}")

    for cond_name in sorted(retrievers.keys()):
        retriever = retrievers[cond_name]
        # 记录 LLM 调用前的 count
        pre_calls = llm.call_count

        try:
            results = retriever.retrieve(query, top_k=2)
        except Exception as e:
            print(f"\n[{cond_name}] ERROR: {e}")
            continue

        llm_calls = llm.call_count - pre_calls

        print(f"\n[{cond_name}] retriever.name={retriever.name!r}, "
              f"LLM 调用 {llm_calls} 次")
        if not results:
            print(f"  返回: 空列表")
        else:
            for i, node in enumerate(results, 1):
                print(f"  Top {i}: {node.id} - {node.title}")

    # === 验证关键性质 ===
    print(f"\n{'='*78}")
    print(f"关键性质验证 (Phase 4.1 Week 3 实验前置 sanity check):")
    print(f"{'='*78}")

    # A: 必须空
    a_results = retrievers["A_null"].retrieve(query, top_k=2)
    print(f"\nA (null) 应返回空: {'✅' if len(a_results) == 0 else '❌'} "
          f"(实际 {len(a_results)})")

    # C: BM25 应能召回相关节点 (含 'lattice' / 'path' 字面词的 lattice_path_counting)
    c_results = retrievers["C_bm25_only"].retrieve(query, top_k=3)
    c_has_lattice = any(n.id == "lattice_path_counting" for n in c_results)
    print(f"C (BM25) 应召回 lattice_path_counting (含 'lattice' 'path'): "
          f"{'✅' if c_has_lattice else '❌'}")

    # F: 不应包含与 query 最相关的节点
    f_results = retrievers["F_irrelevant"].retrieve(query, top_k=2)
    f_has_lattice = any(n.id == "lattice_path_counting" for n in f_results)
    print(f"F (irrelevant) 不应含 lattice_path_counting: "
          f"{'✅' if not f_has_lattice else '❌'}")

    # B (hybrid) 应至少返回一些结果, 且 LLM 被调用
    b_results = retrievers["B_hybrid"].retrieve(query, top_k=2)
    print(f"B (hybrid) 应返回非空 + 调用 LLM (用 Mock 已确定): "
          f"{'✅' if len(b_results) > 0 else '❌'}")

    # D / E 应调用 LLM
    pre = llm.call_count
    retrievers["D_llm_only"].retrieve(query, top_k=2)
    d_calls = llm.call_count - pre
    print(f"D (LLM-only) 应调用 LLM 1 次: {'✅' if d_calls == 1 else '❌'} "
          f"(实际 {d_calls})")

    pre = llm.call_count
    retrievers["E_tree_only"].retrieve(query, top_k=2)
    e_calls = llm.call_count - pre
    print(f"E (Tree-only) 应调用 LLM 1 次: {'✅' if e_calls == 1 else '❌'} "
          f"(实际 {e_calls})")

    # === Bm25 索引内容验证 (用户决策 D-2) ===
    print(f"\n{'='*78}")
    print(f"BM25 索引字段验证 (用户决策 D-2: worked_examples 不进 BM25):")
    print(f"{'='*78}")

    bm25_retriever = retrievers["C_bm25_only"]
    lattice_node = tree.get_node("lattice_path_counting")
    index_text = lattice_node.bm25_index_text()

    # worked_example 独有词 (不在 key_facts)
    in_index_problem = "(0,0) to (4,3)" in index_text
    in_index_step = "monotone path through" in index_text
    in_index_insight = "Always reduce by canceling" in index_text

    print(f"  worked_example.problem '(0,0) to (4,3)' 进 BM25 index: "
          f"{'❌ (期待 False)' if in_index_problem else '✅ (期待 False)'}")
    print(f"  worked_example.solution_steps 进 BM25 index: "
          f"{'❌' if in_index_step else '✅'}")
    print(f"  worked_example.key_insight 进 BM25 index: "
          f"{'❌' if in_index_insight else '✅'}")

    # === Inject prompt 验证 (T-3.7) ===
    llm_inject = lattice_node.llm_inject_text()
    inject_has_examples = "(0,0) to (4,3)" in llm_inject
    print(f"\n  worked_examples 进 LLM inject (T-3.7): "
          f"{'✅' if inject_has_examples else '❌'}")

    # === 总结 ===
    print(f"\n{'='*78}")
    print(f"Demo 完成. LLM 总调用次数: {llm.call_count}")
    print(f"")
    print(f"接下来 Phase 4.1 Week 2-3 工作:")
    print(f"  Week 2: builders.py (Claude API 自动生成 nodes) + corpus 建设")
    print(f"  Week 3: 替换 DemoLLM 为 Claude API callable, 跑 100 题 × 6 conditions")
    print(f"          预算: 600 generations, ~2-3 天 (R1-Distill 4-bit)")
    print(f"{'='*78}")


if __name__ == "__main__":
    demo()
