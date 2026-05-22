"""
knowledge_tree/localizer.py
============================

Phase 4.3 Day 9: 两阶段定位 (LLM localization → generation).

动机 (Day 8 实证):
  - graph_expanded 把 oracle 拉进 candidate (func_hit 6/12, top8)
  - 但 R1 mislocalization 5/12: oracle 在 candidate 里, R1 却改错 function
    (django-14411 铁证: oracle rank 8 进了 candidate, R1 仍 mislocalize)
  - 瓶颈从 "retrieval 召回" 转到 "在 candidate 里选对 function"

两阶段设计 (用户提议, 接近最初 LLM 推理导航):
  Stage 1 (localization): 给 LLM 看 N 个候选 (10-20) 的签名+简介 (省 token),
                          让它选出最相关的 K 个 (default 3).
  Stage 2 (generation): 用选中的 K 个的完整 source_code 喂 anchor prompt 生成 patch.

与现有 LLMRetriever 区别:
  - LLMRetriever 是为数学概念写的 (prompt 说 math problem)
  - 本模块为 SWE-bench bug localization 写, prompt 强调 "which function contains the bug"
  - 输入是已排序的 candidate (graph_expanded 输出), 不是从全 KTF 选

PROTO 关联:
  - PROTO-7.37 (verify LLM 输出来源): localization 结果必须是 candidate 里的 id, 不接受幻觉 id
  - 接近最初设计的 TreeNavigationRetriever / HybridRetriever 的 rerank 阶段
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Callable

logger = logging.getLogger(__name__)


LOCALIZE_PROMPT_TEMPLATE = """You are a bug localization expert. Given a bug report and a list of candidate functions from the codebase, identify which functions most likely need to be MODIFIED to fix the bug.

## Bug Report
{problem_statement}

## Candidate Functions (retrieved from codebase)
{candidates_listing}

## Instructions
Analyze the bug report and select the {select_k} candidate function(s) that most likely contain the code that needs to be CHANGED to fix this bug.

Important:
- Select functions where the actual FIX should go, not just functions related to the topic.
- The bug report may mention symptoms in one function but the fix may belong in another (e.g. a helper it calls, or a sibling method in the same class).
- If a candidate calls another candidate, consider which one holds the buggy logic.
- Reproducer/test code in the bug report is illustrative; do NOT select functions just because they appear in example code.

Respond ONLY with a JSON object in this exact format (no other text, no markdown).
Use the EXACT ID string shown after "ID=" for each function (a long identifier like "astropy_io_fits_fitsrec_py___scale_back_ascii"), NOT a number or position:
{{"selected_ids": ["<exact ID= string>", "<exact ID= string>"], "reasoning": "one short sentence why"}}"""


@dataclass
class LocalizationResult:
    """Stage 1 定位结果."""
    selected_ids: list[str] = field(default_factory=list)
    reasoning: str = ""
    raw_response: str = ""
    n_candidates: int = 0
    fell_back: bool = False  # True 若 LLM 失败, 回退到原排序


def _build_candidates_listing(candidates: list, max_sig_chars: int = 200) -> str:
    """构造候选列表展示 (id + qualified_name + signature + 1-line def, 省 token).

    不含完整 source_code (那是 Stage 2 的事), 只给 LLM 足够信息做选择.
    """
    lines = []
    for node in candidates:
        dm = node.domain_metadata
        qn = dm.get('qualified_name', node.id)
        sig = dm.get('signature', '') or ''
        if len(sig) > max_sig_chars:
            sig = sig[:max_sig_chars] + '...'
        file = dm.get('file', '?')
        # 1-line definition (取首行)
        defn = (node.definition or '').split('\n')[0][:120]
        # 关键 (Day 9 fix): 不用前导序号 (1. 2. ...), LLM 会把序号当 id 返回.
        # 真实 id 用 ID=<...> 显式标注, 是唯一可选标识.
        lines.append(
            f"- ID={node.id}\n"
            f"  function: {qn}  [{file}]\n"
            f"  signature: {sig}\n"
            f"  summary: {defn}"
        )
    return "\n".join(lines)


def _parse_localization_response(
    response: str,
    valid_ids: set[str],
    ordered_ids: Optional[list[str]] = None,
) -> tuple[list[str], str]:
    """解析 LLM localization 响应, 抽 selected_ids + reasoning.

    容错:
      - 去 markdown fence
      - 只接受 valid_ids 中的 id (PROTO-7.37: 不接受幻觉 id)
      - 序号兜底 (Day 9 实证): LLM 常返回 "1" / "id1" / "13" 这种 1-based 序号
        而非真实 id. 若 ordered_ids 提供, 把序号映射回真实 id.
    """
    # 去 ```json fence
    cleaned = re.sub(r'```(?:json)?\s*', '', response).replace('```', '').strip()
    # 找第一个 { ... } JSON 块
    m = re.search(r'\{[\s\S]*\}', cleaned)
    if not m:
        logger.warning("Localization: no JSON found in response")
        return [], ""
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        logger.warning("Localization: JSON parse failed")
        return [], ""

    raw_ids = data.get("selected_ids", [])
    reasoning = data.get("reasoning", "") or ""
    if not isinstance(raw_ids, list):
        logger.warning("Localization: selected_ids not a list: %r", raw_ids)
        return [], reasoning

    def _resolve_ordinal(token: str) -> Optional[str]:
        """把序号 token 映射回真实 id (1-based).

        处理: "1", "id1", "ID=1", "#1", "13" 等. 序号是 1-based (LLM 习惯).
        """
        if not ordered_ids:
            return None
        # 抽 token 里的数字
        nm = re.search(r'\d+', str(token))
        if not nm:
            return None
        n = int(nm.group(0))
        # 1-based → index n-1
        if 1 <= n <= len(ordered_ids):
            return ordered_ids[n - 1]
        return None

    valid = []
    for nid in raw_ids:
        nid_str = str(nid)
        if nid_str in valid_ids:
            valid.append(nid_str)
            continue
        # 兜底: 试序号映射
        resolved = _resolve_ordinal(nid_str)
        if resolved and resolved in valid_ids and resolved not in valid:
            logger.info("Localization: mapped ordinal %r → %s", nid_str, resolved)
            valid.append(resolved)
        else:
            logger.warning("Localization: LLM selected non-existent id %r, skip", nid_str)
    return valid, reasoning


def localize(
    problem_statement: str,
    candidates: list,
    llm_callable: Callable[[str], str],
    select_k: int = 3,
    prompt_template: Optional[str] = None,
) -> LocalizationResult:
    """Stage 1: LLM 从 candidates 中选出最可能含 bug 的 select_k 个.

    Args:
        problem_statement: bug 报告
        candidates: list of KnowledgeNode (graph_expanded 输出, 已排序)
        llm_callable: LLM 调用 (prompt -> response)
        select_k: 选几个 (default 3)

    Returns:
        LocalizationResult (selected_ids 保持 LLM 选择顺序; 失败时 fell_back=True
        并回退到 candidates 前 select_k 个的 id)
    """
    result = LocalizationResult(n_candidates=len(candidates))

    if not candidates:
        result.fell_back = True
        return result

    # candidates 少于 select_k, 直接全选 (不必问 LLM)
    if len(candidates) <= select_k:
        result.selected_ids = [n.id for n in candidates]
        result.reasoning = "all candidates selected (count <= select_k)"
        return result

    valid_ids = {n.id for n in candidates}
    ordered_ids = [n.id for n in candidates]
    listing = _build_candidates_listing(candidates)
    prompt = (prompt_template or LOCALIZE_PROMPT_TEMPLATE).format(
        problem_statement=problem_statement,
        candidates_listing=listing,
        select_k=select_k,
    )

    try:
        response = llm_callable(prompt)
    except Exception as e:
        logger.warning("Localization LLM call failed: %s, fallback to top-%d", e, select_k)
        result.selected_ids = [n.id for n in candidates[:select_k]]
        result.fell_back = True
        return result

    result.raw_response = response
    selected, reasoning = _parse_localization_response(response, valid_ids, ordered_ids)
    result.reasoning = reasoning

    if not selected:
        # LLM 没选出有效 id, 回退到原排序 top-select_k
        logger.warning("Localization: no valid selection, fallback to top-%d", select_k)
        result.selected_ids = [n.id for n in candidates[:select_k]]
        result.fell_back = True
        return result

    result.selected_ids = selected[:select_k]
    return result


def reorder_by_localization(
    candidates: list,
    localization: LocalizationResult,
) -> list:
    """按 localization 结果重排 candidates: 选中的在前 (保持 LLM 顺序), 其余在后.

    这样 Stage 2 generation 用前 K 个 (选中的) 喂 anchor prompt,
    但保留其余作为 fallback (anchor real-file fallback 仍可用).
    """
    selected_set = set(localization.selected_ids)
    id_to_node = {n.id: n for n in candidates}

    # 选中的按 LLM 顺序在前
    reordered = []
    for nid in localization.selected_ids:
        if nid in id_to_node:
            reordered.append(id_to_node[nid])
    # 其余保持原顺序补后
    for n in candidates:
        if n.id not in selected_set:
            reordered.append(n)
    return reordered
