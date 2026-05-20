"""
knowledge_tree/anchor_diff.py
=================================

Phase 4.3 Day 6 Step 2: Anchor-based generation post-processor.

设计动机 (基于 O-D6-30, O-D6-32 实证):
  R1 在 unified diff format 上有 LLM counting failure:
    - content (代码片段) 抄写精确 ✓
    - line number + indentation (metadata) 总是错 ✗
  
  策略: 让 R1 仅输出 BEFORE/AFTER pairs (语义信息),
        程序用 difflib + retrieved 元数据合成正确 line number 的 unified diff.

输入:
  R1 response 文本 (含 BEFORE: ... AFTER: ... blocks)
  retrieved nodes (含 source_code / file path / start_line)
  原 repo path (用于读真实源文件做精确 line 定位)

输出:
  合法 unified diff (line number 由程序算)

PROTO 关联:
  PROTO-7.18 (silent failure 警告): BEFORE 找不到时 logger.warning
  PROTO-7.27 (verify LLM output): BEFORE substring search 是 LLM 输出的 deterministic verify
  PROTO-7.32 (verify before propose): 抽 anchor 后必先 verify 才合成 diff
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class AnchorPair:
    """单个 BEFORE/AFTER 对.
    
    raw_before / raw_after: R1 原始输出 (可能有 indent 错)
    matched_file: 程序找到的真实文件路径 (相对 repo root)
    matched_start_line: 真实源中 BEFORE 第一行的 line number (1-indexed)
    aligned_before: indent-corrected BEFORE (匹配真实源)
    aligned_after: indent-corrected AFTER (与 aligned_before 同 indent)
    match_status: 'exact' | 'indent_corrected' | 'not_found'
    """
    raw_before: str
    raw_after: str
    matched_file: Optional[str] = None
    matched_start_line: Optional[int] = None
    aligned_before: Optional[str] = None
    aligned_after: Optional[str] = None
    match_status: str = "not_found"


# ============================================================================
# Parse BEFORE/AFTER blocks from R1 response
# ============================================================================

# 格式 1 (设计目标): BEFORE: \n```python\n<code>\n```\nAFTER: \n```python\n<code>\n```
_BLOCK_PATTERN_STANDARD = re.compile(
    r"BEFORE\s*:?\s*\n+```(?:python|py|diff)?\s*\n(?P<before>[\s\S]*?)\n```\s*\n+"
    r"AFTER\s*:?\s*\n+```(?:python|py|diff)?\s*\n(?P<after>[\s\S]*?)\n```",
    re.IGNORECASE,
)

# 格式 2 (R1 自然倾向, 实测): 单 fenced block 内用 # BEFORE / # AFTER 注释分隔
# ```python
# # BEFORE
# <before code>
# # AFTER
# <after code>
# ```
_BLOCK_PATTERN_INLINE = re.compile(
    r"```(?:python|py|diff)?\s*\n"
    r"(?:#\s*)?BEFORE\s*\n"
    r"(?P<before>[\s\S]*?)\n"
    r"(?:#\s*)?AFTER\s*\n"
    r"(?P<after>[\s\S]*?)\n"
    r"```",
    re.IGNORECASE,
)

# 格式 3 (XML 风格, prompt 强化用): <BEFORE>...</BEFORE> <AFTER>...</AFTER>
_BLOCK_PATTERN_XML = re.compile(
    r"<BEFORE>\s*\n?(?P<before>[\s\S]*?)\n?\s*</BEFORE>\s*\n?\s*"
    r"<AFTER>\s*\n?(?P<after>[\s\S]*?)\n?\s*</AFTER>",
    re.IGNORECASE,
)


# 格式 4 (R1 自然倾向, hard 题实测): 给完整新函数定义, 隐含 BEFORE = 同名函数现有 body
# ```python
# def my_func(self, args):
#     ...new implementation...
# ```
# Parser 抽 function name + new body, 调用方需要在 retrieved 中找同名函数当 BEFORE.
_FULL_FUNC_PATTERN = re.compile(
    r"```(?:python|py)?\s*\n"
    r"(?P<body>def\s+(?P<name>\w+)\s*\([\s\S]*?)\n```",
    re.IGNORECASE,
)


def parse_anchor_response(response: str) -> list[AnchorPair]:
    """从 R1 response 抽 BEFORE/AFTER pairs.
    
    支持 4 种格式 (实测 R1 倾向于格式 2 和 4):
      1. 标准: BEFORE:\\n```...```\\nAFTER:\\n```...``` (prompt 设计目标)
      2. R1 自然 (注释式): ```...\\n# BEFORE\\n<code>\\n# AFTER\\n<code>\\n``` (实测 django medium)
      3. XML: <BEFORE>...</BEFORE><AFTER>...</AFTER> (prompt 强化用)
      4. 完整函数替换 (hard 题实测): ```python\\ndef func_name(...): ...new body...\\n```
         BEFORE 留空, 由调用方 (response_to_unified_diff) 在 retrieved 找同名函数解析.
    
    Returns: 列表 of AnchorPair (match_status='not_found' 初始)
    """
    pairs = []
    
    # 先按格式 1 尝试 (最严格)
    for m in _BLOCK_PATTERN_STANDARD.finditer(response):
        before = m.group('before').rstrip('\n')
        after = m.group('after').rstrip('\n')
        if before.strip() and after.strip():
            pairs.append(AnchorPair(raw_before=before, raw_after=after))
    
    # 如果格式 1 没找到, 试格式 2 (R1 自然格式, 注释分隔)
    if not pairs:
        for m in _BLOCK_PATTERN_INLINE.finditer(response):
            before = m.group('before').rstrip('\n')
            after = m.group('after').rstrip('\n')
            if before.strip() and after.strip():
                pairs.append(AnchorPair(raw_before=before, raw_after=after))
    
    # 如果还没找到, 试格式 3 (XML)
    if not pairs:
        for m in _BLOCK_PATTERN_XML.finditer(response):
            before = m.group('before').rstrip('\n')
            after = m.group('after').rstrip('\n')
            if before.strip() and after.strip():
                pairs.append(AnchorPair(raw_before=before, raw_after=after))
    
    # 如果仍没找到, 试格式 4 (完整函数替换)
    # 注意: 这个模式可能匹配 BEFORE/AFTER 各自的 ```python``` 块, 所以仅在前 3 格式都失败时尝试
    if not pairs:
        for m in _FULL_FUNC_PATTERN.finditer(response):
            new_body = m.group('body').rstrip('\n')
            func_name = m.group('name')
            # 跳过明显是 method 调用 (如 def 后跟 . 不是空格)
            if not func_name or len(new_body) < 30:  # 太短的函数可能是误抽
                continue
            # AnchorPair: raw_before 留空字符串 + 函数名 marker, AFTER 是新 body
            # 在 hint 字段保存函数名, 让下游 response_to_unified_diff 处理
            pair = AnchorPair(
                raw_before="",   # 占位符, 由 enrich_full_func_pairs 填充
                raw_after=new_body,
                match_status="needs_function_lookup",  # 特殊标记
            )
            # 用 matched_file 字段临时存 function name (奇怪但简单)
            pair.matched_file = f"__lookup_function__:{func_name}"
            pairs.append(pair)
    
    if not pairs:
        logger.warning("No BEFORE/AFTER block found in response (tried 4 formats)")
    return pairs


def enrich_full_func_pairs(
    pairs: list[AnchorPair],
    retrieved_nodes: list,
) -> list[AnchorPair]:
    """处理"完整函数替换"情况: 用 retrieved 中同名函数的真实 body 作为 BEFORE.
    
    两种触发场景:
      A. 格式 4 直接命中 (parse_anchor_response 已标 'needs_function_lookup')
         R1 输出只有完整新函数, 无 BEFORE.
      B. 格式 1/2/3 命中, 但 AFTER 是完整 def 函数 (本函数实证: R1 BEFORE 经常漏 docstring)
         R1 给了 BEFORE, 但不完整. 信任 AFTER 抽函数名, 用 retrieved 覆盖 BEFORE.
    
    格式 4 关键 indent 处理:
      retrieved 中的 source_code 是真实 indent (e.g. method 4 indent),
      但 R1 给的 AFTER 通常是 0 indent (R1 习惯).
      enrich 在此**预 align** raw_after 到 raw_before 的 indent.
    """
    enriched = []
    for p in pairs:
        # 决定是否走"完整函数替换"路径
        func_name = None
        
        if p.match_status == "needs_function_lookup":
            # 场景 A: parse 时已标 (来自格式 4)
            if p.matched_file and p.matched_file.startswith("__lookup_function__:"):
                func_name = p.matched_file.split(":", 1)[1]
        else:
            # 场景 B: 格式 1/2/3 命中, 但 AFTER 是完整 def
            # 检测 AFTER 第一非空行是否为 "def <name>(...)"
            after_stripped = p.raw_after.lstrip()
            m = re.match(r'def\s+(\w+)\s*\(', after_stripped)
            if m and len(p.raw_after) >= 30:  # 够长才考虑函数替换
                func_name = m.group(1)
                # 检查 BEFORE 是否也是 def 同名函数 (R1 给的 BEFORE 可能不完整)
                before_stripped = p.raw_before.lstrip()
                bm = re.match(r'def\s+(\w+)\s*\(', before_stripped)
                if not bm or bm.group(1) != func_name:
                    # BEFORE 不是同名 def, 跳过 (可能 R1 改的不是完整函数)
                    func_name = None
        
        if not func_name:
            enriched.append(p)
            continue
        
        # 在 retrieved nodes 找同名函数
        found = False
        for node in retrieved_nodes:
            sc = getattr(node, 'source_code', None)
            if not sc:
                continue
            sc_stripped = sc.lstrip()
            if not (sc_stripped.startswith(f"def {func_name}(") 
                    or sc_stripped.startswith(f"def {func_name} (")):
                continue
            
            # 找到了
            # 计算 retrieved source_code 的 base indent (method 通常 4)
            _, sc_indent = _strip_uniform_indent(sc)
            # R1 raw_after 通常 0 indent. 把它 re-indent 到 sc_indent.
            _, after_indent = _strip_uniform_indent(p.raw_after)
            indent_delta = sc_indent - after_indent
            if indent_delta > 0:
                prefix = ' ' * indent_delta
                aligned_after_raw = '\n'.join(
                    prefix + l if l.strip() else l 
                    for l in p.raw_after.split('\n')
                )
            elif indent_delta < 0:
                n = -indent_delta
                aligned_after_raw = '\n'.join(
                    l[n:] if l.startswith(' ' * n) else l 
                    for l in p.raw_after.split('\n')
                )
            else:
                aligned_after_raw = p.raw_after
            
            # 重置 pair: raw_before=完整 source_code (覆盖 R1 给的不完整 BEFORE),
            # raw_after=预 align 的版本
            new_pair = AnchorPair(
                raw_before=sc,
                raw_after=aligned_after_raw,
                match_status="not_found",  # 后续 locate_anchor 会修
            )
            enriched.append(new_pair)
            found = True
            break
        
        if not found:
            # 同名函数不在 retrieved
            logger.warning(
                f"Function replacement: function '{func_name}' not in any retrieved.source_code "
                f"(scenario={p.match_status})"
            )
            # 保留原 pair (可能 R1 给的 BEFORE 在真实文件中 fuzzy match 命中, 让 locate_anchor 试)
            # 但如果是场景 A (needs_function_lookup), raw_before 是空字符串, locate_anchor 会 skip
            if p.match_status == "needs_function_lookup":
                p.match_status = "not_found"
                p.matched_file = None  # 清掉 marker
            enriched.append(p)
    
    return enriched


# ============================================================================
# Indent-tolerant substring search
# ============================================================================

def _strip_uniform_indent(text: str) -> tuple[str, int]:
    """删除每行共有的 leading indent. 返回 (stripped, indent_size)."""
    lines = text.split('\n')
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return text, 0
    min_indent = min(len(l) - len(l.lstrip(' ')) for l in non_empty)
    if min_indent == 0:
        return text, 0
    stripped = '\n'.join(l[min_indent:] if l.startswith(' ' * min_indent) else l for l in lines)
    return stripped, min_indent


def _reindent(text: str, target_indent: int) -> str:
    """给每行加 target_indent 个空格 (空行除外)."""
    prefix = ' ' * target_indent
    return '\n'.join(prefix + l if l.strip() else l for l in text.split('\n'))


def find_anchor_in_source(
    raw_before: str,
    source_text: str,
) -> tuple[Optional[int], Optional[str], str]:
    """在 source_text 中查找 raw_before, 容忍 indent 差异 + R1 行级错拼.
    
    搜索顺序:
      Stage 1: byte-exact match (必须行起点对齐)
      Stage 2: indent-corrected match (strip uniform indent → re-indent to source)
      Stage 3 (Phase 4.3 Day 6 新): fuzzy line-anchor match
         适配 R1 拼接/截断 BEFORE 的情况:
         - 找 BEFORE 中"unique anchor line" (非注释, 非常见 keyword)
         - 在 source 中定位该 anchor line
         - 检查 anchor line 周围的 source 行能否拼出"R1 意图" (BEFORE 其他非注释行)
         - 若可, 返回 source 中真实的连续行作为 aligned_before
    
    Returns:
        (start_byte_offset, aligned_before, match_status)
        match_status: 'exact' | 'indent_corrected' | 'fuzzy_line_anchor' | 'not_found'
    """
    # === Stage 1: byte-exact match (行起点对齐) ===
    idx = source_text.find(raw_before)
    while idx >= 0:
        if idx == 0 or source_text[idx - 1] == '\n':
            return idx, raw_before, 'exact'
        idx = source_text.find(raw_before, idx + 1)
    
    # === Stage 2: indent-corrected match ===
    stripped_before, before_indent = _strip_uniform_indent(raw_before)
    
    source_lines = source_text.split('\n')
    stripped_lines = stripped_before.split('\n')
    
    if not stripped_lines or not stripped_lines[0].strip():
        return None, None, 'not_found'
    
    first_anchor = stripped_lines[0].lstrip()
    if not first_anchor:
        return None, None, 'not_found'
    
    for i, sline in enumerate(source_lines):
        sline_stripped = sline.lstrip(' ')
        if not (sline_stripped == first_anchor or sline_stripped.startswith(first_anchor)):
            continue
        line_indent = len(sline) - len(sline.lstrip(' '))
        candidate = _reindent(stripped_before, line_indent)
        candidate_lines = candidate.split('\n')
        match = True
        for j, cl in enumerate(candidate_lines):
            si = i + j
            if si >= len(source_lines):
                match = False
                break
            if cl != source_lines[si]:
                if cl.rstrip() != source_lines[si].rstrip():
                    match = False
                    break
        if match:
            byte_offset = sum(len(l) + 1 for l in source_lines[:i])
            return byte_offset, candidate, 'indent_corrected'
    
    # === Stage 3: fuzzy line-anchor (R1 拼接/截断 BEFORE 的容错) ===
    return _fuzzy_line_anchor_match(raw_before, source_text)


def _is_unique_anchor_line(line: str) -> bool:
    """判断 line 是否适合做 anchor (含具体 token, 非纯注释, 长度足够)."""
    stripped = line.strip()
    if not stripped:
        return False
    # 排除纯注释
    if stripped.startswith('#'):
        return False
    # 排除常见短关键词行
    if stripped in ('pass', 'continue', 'break', 'return', 'else:', 'try:', 'except:'):
        return False
    # 排除单变量行 (太通用)
    if len(stripped) < 15:
        return False
    return True


def _fuzzy_line_anchor_match(
    raw_before: str,
    source_text: str,
) -> tuple[Optional[int], Optional[str], str]:
    """Stage 3: R1 BEFORE 中可能含错拼行 / 缺行 / 多行的情况.
    
    策略:
      1. 抽 BEFORE 中所有 "unique anchor line" (含具体 token, 非注释)
      2. 选最长的 anchor line (最 unique)
      3. 在 source 中找该 line (容忍 indent + 容忍连接到下一行)
      4. 命中后, 返回 source 中匹配的**单行** (作为 aligned_before)
    
    注: 此 Stage 是 lossy - aligned_before 可能只是 BEFORE 的一部分,
    不保证 R1 所有 BEFORE 行都在 source 中. AFTER 应用时需要相应单行替换.
    返回 match_status='fuzzy_line_anchor' 警示 caller.
    """
    before_lines = raw_before.split('\n')
    anchor_candidates = [l for l in before_lines if _is_unique_anchor_line(l)]
    if not anchor_candidates:
        return None, None, 'not_found'
    
    # 选最长的 anchor line (最 unique, 不易 false match)
    anchor_line = max(anchor_candidates, key=lambda l: len(l.strip()))
    anchor_stripped = anchor_line.lstrip()
    
    # 在 source 中找匹配此 stripped anchor 的行 (行起点 + lstrip 后等)
    source_lines = source_text.split('\n')
    matches = []
    for i, sline in enumerate(source_lines):
        sline_stripped = sline.lstrip(' ')
        if sline_stripped == anchor_stripped:
            matches.append(i)
    
    if not matches:
        return None, None, 'not_found'
    
    if len(matches) > 1:
        logger.warning(
            f"Fuzzy anchor matched {len(matches)} times in source, "
            f"using first occurrence. anchor={anchor_stripped[:60]!r}"
        )
    
    # 命中! 返回该行作为 aligned_before
    matched_idx = matches[0]
    matched_line = source_lines[matched_idx]
    byte_offset = sum(len(l) + 1 for l in source_lines[:matched_idx])
    
    logger.info(
        f"Fuzzy line-anchor match: source line {matched_idx + 1}: {matched_line[:80]!r}"
    )
    return byte_offset, matched_line, 'fuzzy_line_anchor'


# ============================================================================
# Locate anchor in retrieved nodes + real source file
# ============================================================================

def locate_anchor(
    raw_before: str,
    retrieved_nodes: list,  # list of KnowledgeNode (含 source_code)
    repo_root: Path,
    search_real_files: bool = True,
) -> tuple[Optional[str], Optional[int], Optional[str], Optional[str], str]:
    """在 retrieved nodes 的 source_code 中找 anchor, 然后映射到真实文件 line.
    
    搜索顺序 (Phase 4.3 Day 6 增强):
      1. retrieved_nodes 的 source_code (function-level inject)
      2. (fallback) retrieved_nodes 引用的真实源文件 (case: problem_statement 含代码或同文件其他函数)
    
    Args:
        raw_before: R1 输出的 BEFORE 代码
        retrieved_nodes: list of KnowledgeNode
        repo_root: 真实 repo 路径 (用于 fallback 搜真实文件)
        search_real_files: 是否在 retrieved 没命中时, fallback 到真实源文件 (默认 True)
    
    Returns:
        (matched_file, matched_start_line, aligned_before, aligned_after_template, match_status)
    """
    # === Stage 1: 在 retrieved nodes 的 source_code 中找 ===
    for node in retrieved_nodes:
        if not getattr(node, 'source_code', None):
            continue
        source_code = node.source_code
        offset, aligned_before, status = find_anchor_in_source(raw_before, source_code)
        if status == 'not_found':
            continue
        # 拿 node 的 file + start_line, 算出 real 文件 line
        file_path = node.domain_metadata.get('file')
        start_line = node.domain_metadata.get('start_line', 1)
        if not file_path:
            logger.warning(f"Node {node.id} missing 'file' metadata, skip")
            continue
        # byte offset → line offset (in source_code)
        line_offset = source_code[:offset].count('\n')
        real_line = start_line + line_offset  # 1-indexed
        return file_path, real_line, aligned_before, None, status
    
    # === Stage 2: Fallback - 在 retrieved nodes 引用的真实源文件中搜 ===
    # 案例:
    #   - problem_statement 含 BEFORE 真实代码, 但 retrieve miss 对应 function
    #   - BEFORE 在同 file 的相邻 function (retrieve 给了同 file 但不同 function)
    if search_real_files:
        # 收集 unique file paths (按 retrieve rank 顺序)
        seen_files = set()
        for node in retrieved_nodes:
            file_path = node.domain_metadata.get('file')
            if not file_path or file_path in seen_files:
                continue
            seen_files.add(file_path)
            real_line, aligned_before, status = locate_anchor_in_file(
                raw_before, repo_root, file_path,
            )
            if status != 'not_found':
                logger.info(
                    f"Anchor found via real-file fallback: file={file_path}, "
                    f"line={real_line}, status={status} "
                    f"(BEFORE not in any retrieved.source_code)"
                )
                return file_path, real_line, aligned_before, None, status
    
    # === Stage 3: 都没找到 ===
    return None, None, None, None, 'not_found'


def locate_anchor_in_file(
    raw_before: str,
    repo_root: Path,
    file_path: str,
) -> tuple[Optional[int], Optional[str], str]:
    """直接在某个文件中找 anchor (用于 retrieved miss 但文件已知的场景)."""
    full_path = repo_root / file_path
    if not full_path.exists():
        return None, None, 'not_found'
    source_text = full_path.read_text(encoding='utf-8', errors='replace')
    offset, aligned_before, status = find_anchor_in_source(raw_before, source_text)
    if status == 'not_found':
        return None, None, 'not_found'
    line_offset = source_text[:offset].count('\n')
    return line_offset + 1, aligned_before, status  # 1-indexed


# ============================================================================
# Synthesize unified diff
# ============================================================================

def synth_unified_diff(
    pairs: list[AnchorPair],
    repo_root: Path,
    context_lines: int = 3,
) -> tuple[str, list[str]]:
    """从 anchored pairs 合成 unified diff.
    
    每个 pair 必须有 matched_file + matched_start_line + aligned_before + aligned_after.
    
    Returns:
        (patch_text, warnings)
    """
    warnings = []
    
    # 按 file 分组
    by_file: dict[str, list[AnchorPair]] = {}
    for p in pairs:
        if p.match_status == 'not_found' or not p.matched_file:
            warnings.append(f"Skip unanchored pair (status={p.match_status})")
            continue
        by_file.setdefault(p.matched_file, []).append(p)
    
    if not by_file:
        return "", warnings + ["No anchored pairs to synth"]
    
    # 每个 file: 读原文 → 按每个 pair 替换 → diff
    diff_parts = []
    for file_path, file_pairs in by_file.items():
        full_path = repo_root / file_path
        if not full_path.exists():
            warnings.append(f"File not found: {file_path}, skip")
            continue
        
        original = full_path.read_text(encoding='utf-8', errors='replace')
        modified = original
        # 按 start_line 降序排序, 从后往前替换 (避免 line shift)
        sorted_pairs = sorted(file_pairs, key=lambda p: p.matched_start_line or 0, reverse=True)
        
        for p in sorted_pairs:
            if not p.aligned_before or p.aligned_after is None:
                warnings.append(f"Pair missing aligned content, skip")
                continue
            # 单次 replace (从 modified 中找 aligned_before, 替换为 aligned_after)
            idx = modified.find(p.aligned_before)
            if idx < 0:
                warnings.append(f"aligned_before disappeared after prior replacement, skip")
                continue
            modified = modified[:idx] + p.aligned_after + modified[idx + len(p.aligned_before):]
        
        # difflib 合成 unified diff
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)
        diff_iter = difflib.unified_diff(
            original_lines, modified_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            n=context_lines,
        )
        diff_text = ''.join(diff_iter)
        if diff_text:
            # 添加 'diff --git' header (difflib 不写)
            diff_text = f"diff --git a/{file_path} b/{file_path}\n" + diff_text
            diff_parts.append(diff_text)
    
    # 合并所有 file diff
    final = '\n'.join(diff_parts)
    # 确保 trailing newline (SWE-bench Issue #145 教训)
    if final and not final.endswith('\n'):
        final += '\n'
    return final, warnings


# ============================================================================
# Aligned AFTER (key step)
# ============================================================================

def align_after(
    raw_after: str,
    raw_before: str,
    aligned_before: str,
) -> str:
    """根据 aligned_before 调整 raw_after 的 indent.
    
    R1 输出的 BEFORE/AFTER 可能 indent 错位, 但二者 indent 一致 (R1 内部相对一致).
    如果 aligned_before 比 raw_before 多/少 indent N, 则 aligned_after = raw_after 加/减 N.
    """
    # 计算 raw_before 和 aligned_before 的 indent 差
    _, raw_indent = _strip_uniform_indent(raw_before)
    _, aligned_indent = _strip_uniform_indent(aligned_before)
    indent_delta = aligned_indent - raw_indent
    
    if indent_delta == 0:
        return raw_after
    
    if indent_delta > 0:
        # 加 indent
        prefix = ' ' * indent_delta
        return '\n'.join(prefix + l if l.strip() else l for l in raw_after.split('\n'))
    else:
        # 减 indent
        n = -indent_delta
        result_lines = []
        for l in raw_after.split('\n'):
            if l.startswith(' ' * n):
                result_lines.append(l[n:])
            else:
                result_lines.append(l)
        return '\n'.join(result_lines)


# ============================================================================
# High-level entry: response → unified diff
# ============================================================================

def response_to_unified_diff(
    response: str,
    retrieved_nodes: list,
    repo_root: Path,
    context_lines: int = 3,
) -> tuple[str, list[AnchorPair], list[str]]:
    """完整流程: R1 response → unified diff.
    
    Returns:
        (patch_text, anchor_pairs, warnings)
        anchor_pairs: 含 match_status 用于诊断
        warnings: 失败/警告信息
    """
    warnings = []
    
    # 1. parse BEFORE/AFTER blocks (含 4 种格式, 含格式 4 full function replacement)
    pairs = parse_anchor_response(response)
    if not pairs:
        return "", [], ["No BEFORE/AFTER blocks in response"]
    
    # 1.5. 处理格式 4 (完整函数替换): 用 retrieved 同名函数当 BEFORE
    pairs = enrich_full_func_pairs(pairs, retrieved_nodes)
    
    # 2. 对每个 pair, 在 retrieved nodes 中找 anchor (Stage 1: retrieved.source_code, Stage 2 fallback: 真实源文件)
    for p in pairs:
        if not p.raw_before:
            # 仍是空 raw_before (格式 4 未找到同名函数)
            warnings.append(f"Empty raw_before for pair (status={p.match_status})")
            continue
        file_path, real_line, aligned_before, _, status = locate_anchor(
            p.raw_before, retrieved_nodes, repo_root,
        )
        if status == 'not_found':
            warnings.append(f"Anchor not found: {p.raw_before[:60]!r}")
            p.match_status = 'not_found'
            continue
        p.matched_file = file_path
        p.matched_start_line = real_line
        p.aligned_before = aligned_before
        p.aligned_after = align_after(p.raw_after, p.raw_before, aligned_before)
        p.match_status = status
    
    # 3. synth diff
    patch_text, synth_warnings = synth_unified_diff(pairs, repo_root, context_lines)
    warnings.extend(synth_warnings)
    
    return patch_text, pairs, warnings
