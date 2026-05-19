# Knowledge Tree Framework (KTF)

> Domain-Agnostic Knowledge Tree Framework for enhancing LLM capability ceiling via structured RAG.

**Status:** Research-in-progress. Phase 4.3 Day 5 complete (SWE-bench Lite sanity).

---

## What is this?

A 4-layer abstraction (`KnowledgeNode` + `TreeBuilder` + `KnowledgeStorage` + `Retriever`) that enables systematic knowledge augmentation for distilled LLMs (e.g. R1-Distill-Qwen-14B, Nemotron-Nano-9B).

**Validated across two domains** (T-3.8 SEALED, framework v3.6):
- Phase 4.1: Mathematics (AIME / MATH-Hard)
- Phase 4.2: Polars 1.0+ code generation (50 tasks, B-F +44pp paired CI SIG)
- Phase 4.3: SWE-bench Lite (in progress, Day 5 sanity)

## Core Design

```
KnowledgeNode (definition + key_facts + worked_examples + common_pitfalls + domain_metadata)
    ↓
TreeBuilder (LLM-based / Python introspection / AST-based)
    ↓
KnowledgeStorage (JSON / SQLite / Neo4j)
    ↓
Retriever (BM25 / Hybrid / Tree Navigation / Self-Knowledge Filter)
    ↓
Generator LLM
```

## Key Findings (SEALED in framework v3.6)

- **T-3.7**: RAG efficacy needs procedural knowledge (worked_examples + prescriptive pitfalls)
- **T-3.9 v2**: H-M 6 prerequisites for RAG utility:
  (i) baseline wrong + (ii) identifiable bottleneck + (iii) model genuinely lacks + 
  (iv) retriever recalls correctly + (v) generator format compliant + (vi) semantic correct
- **T-3.13**: Generator format & capability strongly coupled (R1 67% format failure on SWE-bench)
- **T-3.14**: R1 tends to "modify retrieved function" rather than "find correct location"

## Quick Start

```bash
# Install
git clone https://github.com/wangyichengsh/Knowledge-Tree-Framework.git
cd Knowledge-Tree-Framework
pip install -e .

# Run tests (335 tests)
python -m unittest discover -s tests

# Try Polars benchmark
python scripts/polars_sanity_check.py  # 1-task sanity (PROTO-7.21)
```

## Documents

- [`docs/intelligence_framework_v3_6.md`](docs/intelligence_framework_v3_6.md) — Core findings, SEALED triples, PROTO-7
- [`docs/agi_engineering_architecture_v1_12.md`](docs/agi_engineering_architecture_v1_12.md) — Engineering details, file inventory
- [`docs/SESSION_HANDOFF.md`](docs/SESSION_HANDOFF.md) — For continuing across LLM sessions

## Repository Structure

```
Knowledge-Tree-Framework/
├── knowledge_tree/                # Core KTF (10 files, ~5K lines)
├── tests/                         # 335 unit tests, all pass
├── scripts/                       # Experiment runners
└── docs/                          # Framework, architecture, session handoffs
```

## Phase Progress

- ✅ Phase 4.1 (Mathematics): T-3.4 capability ceiling triage
- ✅ Phase 4.2 (Polars code): T-3.7/3.8/3.9/3.10/3.11/3.12 SEALED
- 🔄 Phase 4.3 (SWE-bench): Day 5 sanity complete, Day 6 Fork B + 10-task pilot

## Hardware

- Tested: NVIDIA RTX 5090 32GB, Ubuntu 24
- Models: R1-Distill-Qwen-14B (4-bit + LoRA), Nemotron-Nano-9B-v2 (4-bit)
- Frameworks: PyTorch, transformers, polars 1.40, tree-sitter 0.25, swebench 4.1.0


