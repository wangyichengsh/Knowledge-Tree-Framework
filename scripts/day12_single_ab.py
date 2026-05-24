#!/usr/bin/env python3
"""
scripts/day12_single_ab.py — 单题富化 A/B: 直接量 oracle 在完整 ranking 的 rank.

比 sweep 更精确: 不只看 top-k 命中, 而是看 oracle 在全部节点里排第几,
能定量看出富化让 rank 前移多少 (rank 311 → 5 这种).

用法:
  python scripts/day12_single_ab.py \
      --ktf /tmp/swe-bench-day10/scikit-learn__scikit-learn-12471/ktf.json \
      --candidates day10_balanced50.json \
      --instance-id scikit-learn__scikit-learn-12471
"""
import argparse, json, re, sys, os
sys.path.insert(0, os.environ.get('PROJECT_ROOT', '.'))


def parse_oracle(patch):
    files = re.findall(r'diff --git a/(\S+)', patch)
    funcs = []
    for line in patch.split('\n'):
        m = re.match(r'@@ .+ @@ ?(.*)', line)
        if m:
            fm = re.match(r'(?:async\s+)?def\s+(\w+)', m.group(1).strip())
            cm = re.match(r'class\s+(\w+)', m.group(1).strip())
            if fm: funcs.append(fm.group(1))
            elif cm: funcs.append(cm.group(1))
    return files, list(dict.fromkeys(funcs))


def oracle_rank(ktf_path, query, oracle_files, oracle_funcs, include_llm_summary):
    from knowledge_tree.storage import JSONStorage
    from knowledge_tree.core import KnowledgeTree
    from knowledge_tree.retrievers import BM25Retriever
    storage = JSONStorage(str(ktf_path), create_if_missing=False)
    tree = KnowledgeTree(storage.list_all())
    bm25 = BM25Retriever(tree, include_llm_summary=include_llm_summary)
    ranked = bm25.get_ranked_with_scores(query, top_k=10000)
    for rank, (node, score) in enumerate(ranked, 1):
        dm = node.domain_metadata or {}
        qn = dm.get('qualified_name', '')
        fn = qn.split('.')[-1]
        nfile = dm.get('file', '')
        if fn in oracle_funcs and any(of in nfile or nfile in of for of in oracle_files if nfile):
            return rank, score, qn
    return None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ktf', required=True)
    ap.add_argument('--candidates', required=True)
    ap.add_argument('--instance-id', required=True)
    args = ap.parse_args()

    cands = json.load(open(args.candidates))
    task = [v for v in cands.values() if v and v.get('instance_id') == args.instance_id][0]
    ofiles, ofuncs = parse_oracle(task['patch'])
    query = task['problem_statement']

    print(f"=== {args.instance_id} ===")
    print(f"oracle functions: {ofuncs}")
    print(f"oracle files: {ofiles}\n")

    r_base, s_base, qn_base = oracle_rank(args.ktf, query, ofiles, ofuncs, False)
    r_enr, s_enr, qn_enr = oracle_rank(args.ktf, query, ofiles, ofuncs, True)

    print(f"{'':20} {'rank':>8} {'score':>10}")
    print(f"{'baseline':20} {str(r_base):>8} {s_base if s_base else 0:>10.3f}")
    print(f"{'enriched':20} {str(r_enr):>8} {s_enr if s_enr else 0:>10.3f}")
    if r_base and r_enr:
        delta = r_base - r_enr
        print(f"\noracle rank 变化: {r_base} → {r_enr} ({'前移' if delta>0 else '后退'} {abs(delta)})")
        print(f"进 top-3? baseline={'是' if r_base<=3 else '否'}, enriched={'是' if r_enr<=3 else '否'}")
    elif r_enr and not r_base:
        print(f"\noracle 从'未召回'→ rank {r_enr} (富化救回!)")
    elif r_base and not r_enr:
        print(f"\n⚠️ oracle 从 rank {r_base} → 未召回 (富化导致退化!)")


if __name__ == '__main__':
    main()
