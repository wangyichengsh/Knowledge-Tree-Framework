# Knowledge Tree Framework (KTF)

> Domain-agnostic knowledge tree for enhancing an LLM's capability ceiling via structured, graph-expanded RAG — applied end-to-end to SWE-bench patch generation.

**Status:** Research-in-progress. Phase 4.3 (SWE-bench Lite) complete through Day 12; summary-enrichment mechanism validated, full-scale enrichment pending a cheaper local model.

---

## What is this?

A 4-layer abstraction (`KnowledgeNode` + `TreeBuilder` + `KnowledgeStorage` + `Retriever`) that turns a codebase (or any knowledge domain) into a retrievable knowledge tree, then feeds the right context to a generator LLM to produce fixes.

On SWE-bench Lite, KTF + Claude Opus 4.7 in a **single-pass** pipeline (retrieve → localize → generate) reaches **48%** resolved on a 33-task cross-repo sample, rising to **61%** with one round of failure-feedback rescue — approaching the Lite leaderboard top (62.7%, which uses 250-turn agents), with a far simpler method.

### Headline results (33-task cross-repo sample)

| Generator (same KTF framework) | Resolved | Notes |
|---|---|---|
| R1-Distill-14B + LoRA (local) | ~1/33 (3%) | local model is the bottleneck |
| Claude Sonnet 4.6 | 9/33 (27%) | |
| **Claude Opus 4.7** | **16/33 (48%)** | func_hit→resolved 63% |
| **Opus 4.7 + rescue** | **20/33 (61%)** | one feedback-driven retry round |

The **func_hit→resolved** metric (oracle function was retrieved *and* the patch resolved) is used to separate genuine framework contribution from the model's memorization of well-known open-source issues (data contamination) — see framework T-3.18.

---

## Core Design

```
KnowledgeNode (definition + key_facts + source_code + domain_metadata[+ llm_summary])
    ↓
TreeBuilder  ── AST-based (structure) │ + LLM enrichment (semantic bridge)
    ↓
KnowledgeStorage (JSON / SQLite)
    ↓
Retriever  ── BM25 seed → GraphExpanded (same_class + call graph, signal-layered rerank)
    ↓
Localizer  ── two-stage: LLM picks top-K from N candidates
    ↓
Generator LLM ── anchor-based diff (LLM emits BEFORE/AFTER, program synthesizes the diff)
```

---

## Usage

### Scenario 1 — Build a KTF from an existing project

Two build modes, trading cost for retrieval power:

**Mode A — Structure only (fast, free, no LLM).** The AST builder extracts each function/method/class with its signature, docstring, source code, and call graph. Index text is built from names, paths, and signatures.

```bash
python -c "
from knowledge_tree.ast_tree_builder import ASTTreeBuilder
from knowledge_tree.storage import JSONStorage

builder = ASTTreeBuilder(include_classes=True, path_prefix='mypackage')
nodes = builder.build_from_repo(
    repo_path='path/to/mypackage',
    file_glob='**/*.py',
    ignore_patterns=['tests/', 'docs/', '__pycache__'],
)
storage = JSONStorage('mypackage_ktf.json', create_if_missing=True, autosave=False)
for n in nodes: storage.save_node(n)
storage.flush()
print(f'built {len(nodes)} nodes')
"
```

**Mode B — Semantic bridge (LLM-enriched).** On top of Mode A, an LLM writes an objective behavior description (`llm_summary`) for each function and a structural description (`class_summary`) for each class. This closes the *vocabulary gap*: bug reports describe *behavior* ("removes unknown categories"), while raw code only exposes *names* (`_transform`). Enrichment lets BM25 match behavioral language.

```bash
# Enrich an existing KTF in place. Model can be local (r1/nemotron) or API (sonnet/haiku).
python scripts/day12_enrich_summary.py \
    --ktf mypackage_ktf.json \
    --model claude_api --claude-model claude-sonnet-4-6
# (this is a "translation" task, not reasoning — a small/local model is usually sufficient)
```

Enrichment is a **safe, one-directional gain**: it moves vocabulary-gap targets up sharply (measured: rank 369→28, 311→25) while functions already retrieved via structural signals stay put (BM25 layer may dip, but the same_class/call rerank layer absorbs it — see T-3.19).

Verify the effect on a single task without spending generation budget:

```bash
python scripts/day12_single_ab.py \
    --ktf mypackage_ktf.json --candidates tasks.json --instance-id <id>
# prints oracle rank baseline vs enriched
```

### Scenario 2 — Generate a diff from a KTF given a requirement

Given a problem statement (bug report / feature request) and a built KTF, the pipeline retrieves relevant functions, optionally has an LLM localize the best candidates, and generates an **anchor-based** patch (the LLM outputs BEFORE/AFTER code blocks; the program locates them in the real file and synthesizes a correct-line-number unified diff — this sidesteps LLM line-number hallucination).

```bash
python scripts/day7_pipeline.py \
    --candidates tasks.json \
    --model claude_api --claude-model claude-opus-4-7 \
    --retriever graph_expanded \
    --seed-k 5 --candidate-k 30 --top-k 3 --max-expansion 40 \
    --localize --retry 2 \
    --include-llm-summary \          # use enriched KTF if you ran Mode B
    --work-dir /tmp/run --skip-build --skip-clone
```

Best-known parameters (from a zero-API retrieval sweep): `seed_k=5, candidate_k=30, max_expansion=40, top_k=3, localize on`. Note the trade-off: larger `candidate_k` raises recall but also raises localization burden (T-3.21).

Per-task outputs land in `<work-dir>/<instance_id>/`: `ktf.json`, `retrieved.json`, `localization.json`, `generated_patch.diff`, `anchor_metadata.json`.

### Scenario 3 — Run the SWE-bench evaluation

```bash
# 1. (optional) pick a balanced cross-repo task set
python scripts/day10_select_balanced.py --total 50 \
    --repos django astropy sympy scikit-learn matplotlib sphinx \
    --output tasks.json

# 2. (optional, zero API cost) sweep retrieval params to find the recall ceiling
python scripts/day10_retrieval_sweep.py --candidates tasks.json \
    --work-dir /tmp/run --sweep-seed-k 3 5 --sweep-candidate-k 15 30 --sweep-max-expansion 20 40

# 3. run the pipeline (Scenario 2) to produce patches, then build predictions
python scripts/day7_step6_prepare.py --candidates tasks.json \
    --patches-dir /tmp/run --output predictions.jsonl

# 4. evaluate with the official harness
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path predictions.jsonl --max_workers 4 --run_id my_run
```

### Scenario 4 — Rescue: a second round on unresolved tasks

After the harness reports `unresolved_ids`, a rescue round re-retrieves with a larger candidate pool, feeds a **pitfall** signal (the failed diff, telling the model "this change didn't pass — reconsider"), and regenerates. Effective on tasks where the oracle was retrieved but the first attempt changed the wrong function; ineffective on retrieval misses (see T-3.22).

```bash
python scripts/day11_rescue.py \
    --candidates tasks.json \
    --harness-result <harness-output>.json \
    --round1-dir /tmp/run \
    --rescue-dir /tmp/rescue \
    --model claude_api --claude-model claude-opus-4-7 \
    --candidate-k 30 --include-class-summary   # class_summary helps structural-decoupled tasks
    # --only-repo <name>  to validate on a single repo first
```

---

## Key Findings (SEALED in framework)

- **T-3.17** — A mechanism's effect is coupled to model capability: the same two-stage localization *helps* Opus (func_hit 5→6) but *hurts* R1 (6→4). Evaluate LLM-in-the-loop mechanisms with the model held as a control variable.
- **T-3.18** — Data contamination: strong models resolve well-known old issues from memory even when retrieval misses. Use **func_hit→resolved** to measure true framework contribution.
- **T-3.19** — Layer structural signals (same_class/call) *above* lexical signals (BM25 score), don't blend them into one score — a low-BM25 oracle that shares a class gets drowned otherwise.
- **T-3.20** — Anchor-based generation: let the LLM emit BEFORE/AFTER content; offload line-number/format computation to a deterministic program.
- **T-3.21** — `candidate_k` is double-edged: it raises recall *and* localization burden; the func_hit-optimal value is not the resolve-optimal value.
- **T-3.22** — Rescue's boundary: failure-feedback retry recovers *generation* failures (wrong function chosen) but not *retrieval* misses (needs semantic/embedding search).
- **T-3.24** — Leaderboard literacy: 70%+ scores are SWE-bench **Verified** (agent, 250-turn, contaminated), not **Lite** single-pass; compare like with like.

Full triples and reasoning chains live in `docs/`.

---

## Repository Structure

```
Knowledge-Tree-Framework/
├── knowledge_tree/                # Core KTF
│   ├── core.py                    # KnowledgeNode / KnowledgeTree (bm25_index_text + enrichment toggles)
│   ├── storage.py                 # JSON / SQLite storage
│   ├── ast_tree_builder.py        # tree-sitter AST → nodes (structure mode)
│   ├── retrievers.py              # BM25 + GraphExpanded (signal-layered rerank)
│   ├── localizer.py               # two-stage LLM localization
│   ├── anchor_diff.py             # BEFORE/AFTER → synthesized unified diff
│   └── claude_api_client.py       # pure-stdlib Claude API callable (auto-adapts to model)
├── scripts/                       # Experiment / pipeline runners (day7 / day10 / day11 / day12)
├── tests/                         # 317 unit tests
└── docs/                          # Framework, architecture, session handoffs
```

## Quick Start

```bash
git clone https://github.com/wangyichengsh/Knowledge-Tree-Framework.git
cd Knowledge-Tree-Framework
pip install -e .

python -m pytest tests/            # 317 tests
```


## Phase Progress

- ✅ Phase 4.1 (Mathematics): capability-ceiling triage
- ✅ Phase 4.2 (Polars code generation): 50 tasks, paired-CI significant uplift
- ✅ Phase 4.3 (SWE-bench Lite): full pipeline — three-model comparison (1→9→16), retrieval optimization (func_hit 33%→58%), rescue mechanism (48%→61%), summary-enrichment mechanism validated

## Hardware / Stack

- Tested: NVIDIA RTX 5090 32GB, Ubuntu 24
- Models: R1-Distill-Qwen-14B (4-bit + LoRA), Nemotron-Nano-9B-v2 (4-bit), Claude API (Sonnet 4.6 / Opus 4.7)
- Frameworks: PyTorch, transformers, tree-sitter 0.25, swebench 4.1.0

