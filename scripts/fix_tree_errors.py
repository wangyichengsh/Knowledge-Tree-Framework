#!/usr/bin/env python3
"""
scripts/fix_tree_errors.py
===========================

Phase 4.1 Week 2 后处理 - 修复 tree.json 中的 worked_examples 错误.

背景 (用户类型一审查 2026-05-12):
  用户独立审查 tree_fixed.json 的 390 例题, 发现 17 错误 (4.6%):
    7 A 类: 数学/逻辑错误 (必修)
    3 B 类: 题面被改写以可解
    7 C 类: 中间步骤瑕疵但答案对
  
  本脚本只修 A 类 (最严重, 数学上确凿错):
    E1 angle_bisector_theorem ex#3: 矛盾自救硬凑
    E2 binomial_distribution ex#2: 幂运算偏大 1.47x
    E3 catalan_numbers ex#3: strictly vs weakly 混淆
    E4 counting_paths_in_grids ex#3: strictly vs weakly 混淆
    E5 extended_law_of_sines ex#1: SSA 二义情况漏解
    E6 isosceles_tetrahedron_properties ex#2: 外接球公式错
    E7 sum_to_product_formulas ex#3: 题面给的恒等式是错的

设计 (PROTO-7.7 错题分诊优先):
  - 用 Claude API 重新生成单个 example, 不重做整个节点
  - 保留原 example 到 domain_metadata.fixed_examples_audit (审计)
  - prompt 含: 节点 title/def/其他 examples (上下文) + error 描述 + recommendation
  - 修复后再让 Claude 自我 verify (dual check)

成本估算:
  每个 example ~$0.01 (Sonnet 4.6 input ~1.5K + output ~1.5K)
  7 examples × $0.01 = ~$0.07 + verify pass × $0.07 = ~$0.14 total

PROTO 关联:
  PROTO-7.4 (实测校准): 基于用户独立审查的 jsonl
  PROTO-7.6 (不基于"应该 work"): 修复后 verify
  PROTO-7.9 (单测 + 实数据 dual validation): mock + 实数据

用法:
  # 干跑 (mock Claude, 看 prompt 长啥样)
  python scripts/fix_tree_errors.py \\
    --tree-json knowledge_tree/docs/math/tree.json \\
    --errors-jsonl tree_fixed_errors.json \\
    --output tree_fixed_v2.json \\
    --dry-run

  # 实跑
  python scripts/fix_tree_errors.py \\
    --tree-json knowledge_tree/docs/math/tree.json \\
    --errors-jsonl tree_fixed_errors.json \\
    --output tree_fixed_v2.json
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_tree.core import KnowledgeNode, WorkedExample
from knowledge_tree.storage import JSONStorage
from knowledge_tree.builders import LLMTreeBuilder
from knowledge_tree.llm_clients import ClaudeCallable

logger = logging.getLogger(__name__)


# ============================================================================
# Prompt 模板
# ============================================================================

FIX_PROMPT_TEMPLATE = """You are fixing a worked_example in a math knowledge tree.

## Context: The node containing the bad example

- Node ID: "{node_id}"
- Title: "{node_title}"
- Definition: {node_definition}

## Other examples in this node (for style consistency)

{other_examples}

## The BAD example to replace

Problem: {bad_problem}
Solution steps: {bad_solution_steps}
Final answer: {bad_final_answer}
Key insight: {bad_key_insight}

## Why this example is bad

Error type: {error_type}
Summary: {error_summary}
Verification: {error_verification}
Recommendation: {error_recommendation}

## Your task

Generate ONE replacement worked_example for this node. Requirements:

1. The new problem MUST illustrate the same concept as the node title
2. The new problem MUST have a verifiable correct solution (no contradictions, no impossible geometry)
3. Use DIFFERENT specific parameters than typical textbook problems
4. The solution_steps MUST be mathematically correct (verify each numerical step)
5. The final_answer MUST be exactly verifiable (run the computation yourself before stating)
6. Match the style/depth of the other examples in this node

For computations:
  - Verify all power/trig/special function values numerically
  - Check geometric existence (triangle inequality, etc.)
  - For "ambiguous case" theorems (SSA, etc.), explicitly handle all valid cases
  - For "strictly vs weakly" conditions, match the problem statement exactly

Output JSON:
{{
  "problem": "problem statement (different parameters from canonical textbook problems)",
  "solution_steps": ["step 1", "step 2", "...", "step N"],
  "final_answer": "the verified correct answer",
  "key_insight": "what pattern/trick this example demonstrates (1-2 sentences)"
}}

Output ONLY the JSON object. No preamble, no markdown code block."""


VERIFY_PROMPT_TEMPLATE = """You are verifying that a worked_example is mathematically correct.

## The example

Problem: {problem}
Solution steps: {solution_steps}
Final answer: {final_answer}

## Your task

Independently verify the math. For each computation, compute it yourself and compare:
  - All arithmetic / algebra steps
  - All special function values (powers, trig, factorials, etc.)
  - Geometric existence conditions (triangle inequality, etc.)
  - "Ambiguous case" coverage (SSA, etc.)
  - "Strictly vs weakly" condition matching

If you find ANY error, output:
{{
  "verified": false,
  "issues": ["specific issue 1", "specific issue 2"]
}}

If everything checks out, output:
{{
  "verified": true,
  "confidence": 0.95
}}

Output ONLY the JSON. No preamble."""


# ============================================================================
# 修复单个 example
# ============================================================================

def format_other_examples(
    node: KnowledgeNode,
    skip_index: int,
) -> str:
    """格式化节点的其他 examples 作为 LLM 上下文."""
    others = []
    for i, ex in enumerate(node.worked_examples):
        if i == skip_index:
            continue
        others.append(
            f"### Example {i + 1}\n"
            f"Problem: {ex.problem}\n"
            f"Final answer: {ex.final_answer}\n"
            f"Key insight: {ex.key_insight}"
        )
    if not others:
        return "(no other examples in this node)"
    return "\n\n".join(others)


def fix_one_example(
    callable_: Any,
    node: KnowledgeNode,
    error: dict,
    verify: bool = True,
) -> tuple[Optional[WorkedExample], dict]:
    """
    修复单个 example.

    Returns:
        (new_example, audit_info)
        new_example: 修复后的 WorkedExample, 或 None (失败)
        audit_info: 修复过程记录 (含 verify 结果, 用于审计)
    """
    ex_idx = error['example_index'] - 1  # report 是 1-indexed
    if ex_idx < 0 or ex_idx >= len(node.worked_examples):
        return None, {
            'status': 'failed_invalid_index',
            'error_id': error['id'],
            'reason': f"example_index {error['example_index']} out of range",
        }

    bad_ex = node.worked_examples[ex_idx]

    # 1. 生成新 example
    other_examples = format_other_examples(node, ex_idx)
    prompt = FIX_PROMPT_TEMPLATE.format(
        node_id=node.id,
        node_title=node.title,
        node_definition=node.definition,
        other_examples=other_examples,
        bad_problem=bad_ex.problem,
        bad_solution_steps=json.dumps(bad_ex.solution_steps, ensure_ascii=False),
        bad_final_answer=bad_ex.final_answer,
        bad_key_insight=bad_ex.key_insight,
        error_type=error.get('type', 'unknown'),
        error_summary=error.get('summary', ''),
        error_verification=error.get('verification', '(not provided)'),
        error_recommendation=error.get('recommendation', '(not provided)'),
    )

    try:
        response = callable_(prompt)
    except Exception as e:
        return None, {
            'status': 'failed_llm_call',
            'error_id': error['id'],
            'reason': str(e),
        }

    # 解析 (复用 LLMTreeBuilder 的 JSON 解析)
    parser_builder = LLMTreeBuilder(callable_)
    parsed = parser_builder._parse_response(response)
    if not parsed:
        return None, {
            'status': 'failed_json_parse',
            'error_id': error['id'],
            'response_preview': response[:300],
        }

    # 构造 WorkedExample
    required_fields = ['problem', 'solution_steps', 'final_answer', 'key_insight']
    missing = [f for f in required_fields if f not in parsed]
    if missing:
        return None, {
            'status': 'failed_missing_fields',
            'error_id': error['id'],
            'missing': missing,
        }

    try:
        new_ex = WorkedExample(
            problem=str(parsed['problem']),
            solution_steps=[str(s) for s in parsed['solution_steps']],
            final_answer=str(parsed['final_answer']),
            key_insight=str(parsed['key_insight']),
        )
    except Exception as e:
        return None, {
            'status': 'failed_construct',
            'error_id': error['id'],
            'reason': str(e),
        }

    audit = {
        'status': 'generated',
        'error_id': error['id'],
        'node_id': node.id,
        'example_index': ex_idx,
        'original_problem': bad_ex.problem,
        'original_answer': bad_ex.final_answer,
        'new_problem': new_ex.problem,
        'new_answer': new_ex.final_answer,
    }

    # 2. Verify (可选)
    if verify:
        verify_prompt = VERIFY_PROMPT_TEMPLATE.format(
            problem=new_ex.problem,
            solution_steps=json.dumps(new_ex.solution_steps, ensure_ascii=False),
            final_answer=new_ex.final_answer,
        )
        try:
            verify_response = callable_(verify_prompt)
            verify_parsed = parser_builder._parse_response(verify_response)
            if verify_parsed:
                audit['verify_result'] = verify_parsed
                if not verify_parsed.get('verified', False):
                    audit['status'] = 'generated_but_verify_failed'
                    logger.warning(
                        "Fix verify failed for %s ex#%d: %s",
                        node.id, error['example_index'],
                        verify_parsed.get('issues', []),
                    )
            else:
                audit['verify_result'] = {'parse_failed': True}
        except Exception as e:
            audit['verify_result'] = {'call_failed': str(e)}

    return new_ex, audit


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tree-json", required=True,
                        help="原 tree.json (含错误的 examples)")
    parser.add_argument("--errors-jsonl", required=True,
                        help="审查报告 JSON (tree_fixed_errors.json)")
    parser.add_argument("--output", required=True,
                        help="修复后输出路径")
    parser.add_argument("--severity", default="A",
                        help="只修哪些严重程度 (A/B/C, 逗号分隔), 默认仅 A")
    parser.add_argument("--no-verify", action="store_true",
                        help="跳过 LLM 自我 verify (省钱省时)")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--max-cost-usd", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true",
                        help="用 mock Claude")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    severities = set(s.strip() for s in args.severity.split(","))
    print("=" * 78)
    print(f"Fix Tree Errors (severity: {severities})")
    print("=" * 78)

    # === 加载 ===
    with open(args.errors_jsonl) as f:
        err_data = json.load(f)
    target_errors = [e for e in err_data['errors'] if e['severity'] in severities]
    print(f"\n目标错误: {len(target_errors)} 个")
    for e in target_errors:
        print(f"  {e['id']} {e['node_id']} ex#{e['example_index']}: {e['type']}")

    storage = JSONStorage(args.tree_json, create_if_missing=False)
    print(f"\nTree: {len(storage)} 节点")

    # === 初始化 Claude ===
    if args.dry_run:
        print("\n=== DRY RUN: 用 MockClaude ===")
        class MockClaude:
            def __init__(self): 
                self.total_calls = 0
                self.total_cost_usd = 0.0
            def __call__(self, prompt):
                self.total_calls += 1
                if "VERIFY" in prompt or "verifying" in prompt:
                    return '{"verified": true, "confidence": 0.95}'
                return json.dumps({
                    "problem": "Mock problem with parameters X=7, Y=4.",
                    "solution_steps": ["Step 1: apply formula", "Step 2: compute"],
                    "final_answer": "42",
                    "key_insight": "Mock insight about the concept",
                })
            def get_stats(self):
                return {'total_calls': self.total_calls, 'total_cost_usd': 0.0,
                        'total_input_tokens': 0, 'total_output_tokens': 0,
                        'total_retries': 0, 'model': 'mock'}
        claude = MockClaude()
    else:
        claude = ClaudeCallable(
            api_key=args.api_key,
            model=args.model,
            max_tokens=2048,
            temperature=0.3,  # 低温度求精准
            verbose=args.verbose,
        )

    # === 逐个修复 ===
    print(f"\n=== 修复阶段 ===")
    audits = []
    success_count = 0
    for i, error in enumerate(target_errors, 1):
        if (not args.dry_run 
                and claude.total_cost_usd >= args.max_cost_usd):
            logger.warning(
                "Budget $%.2f 已用尽, 剩余 %d 个错误跳过",
                args.max_cost_usd, len(target_errors) - i + 1,
            )
            break

        node_id = error['node_id']
        try:
            node = storage.get_node(node_id)
        except KeyError:
            print(f"  [{i}/{len(target_errors)}] ⚠️ 节点 {node_id} 不存在, 跳过")
            audits.append({'status': 'node_not_found', 'error_id': error['id']})
            continue

        print(f"  [{i}/{len(target_errors)}] 修复 {error['id']}: {node_id} ex#{error['example_index']}")

        new_ex, audit = fix_one_example(
            claude, node, error, verify=not args.no_verify,
        )
        audits.append(audit)

        if new_ex is None:
            print(f"    ❌ 失败: {audit.get('status')} {audit.get('reason', '')}")
            continue

        # 替换 example
        ex_idx = error['example_index'] - 1
        old_ex = node.worked_examples[ex_idx]
        new_worked_examples = list(node.worked_examples)
        new_worked_examples[ex_idx] = new_ex
        node.worked_examples = new_worked_examples

        # 审计记录原 example
        if 'fixed_examples_audit' not in node.domain_metadata:
            node.domain_metadata['fixed_examples_audit'] = []
        node.domain_metadata['fixed_examples_audit'].append({
            'error_id': error['id'],
            'example_index': ex_idx,
            'severity': error['severity'],
            'error_type': error['type'],
            'error_summary': error['summary'],
            'original': {
                'problem': old_ex.problem,
                'solution_steps': old_ex.solution_steps,
                'final_answer': old_ex.final_answer,
                'key_insight': old_ex.key_insight,
            },
            'verify_result': audit.get('verify_result'),
            'fix_date': '2026-05-12',
        })

        storage.save_node(node)
        success_count += 1
        verify_marker = ""
        if audit.get('verify_result'):
            if audit['verify_result'].get('verified'):
                verify_marker = " (verified ✓)"
            elif audit['verify_result'].get('verified') is False:
                verify_marker = f" (verify FAIL: {audit['verify_result'].get('issues', [])})"
        print(f"    ✅ 替换: {new_ex.final_answer!r}{verify_marker}")

    storage.flush()

    # === 写出 ===
    if args.output != args.tree_json:
        new_storage = JSONStorage(args.output, create_if_missing=True)
        new_storage.save_nodes(storage.list_all())
        new_storage.flush()

    # === 总结 ===
    print(f"\n=== 总结 ===")
    print(f"  目标错误: {len(target_errors)}")
    print(f"  成功修复: {success_count}")
    print(f"  失败:    {len(target_errors) - success_count}")

    if hasattr(claude, 'get_stats'):
        stats = claude.get_stats()
        print(f"\n  Claude API:")
        print(f"    Calls: {stats.get('total_calls')}")
        print(f"    Cost:  ${stats.get('total_cost_usd', 0):.4f}")

    # 写 audit log
    audit_path = args.output + '.audit.json'
    with open(audit_path, 'w') as f:
        json.dump({
            'audits': audits,
            'success_count': success_count,
            'target_count': len(target_errors),
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  审计日志: {audit_path}")
    print(f"  输出 tree: {args.output}")


if __name__ == "__main__":
    main()
