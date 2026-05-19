"""
knowledge_tree/python_api_builder.py
======================================

Phase 4.2 Stage 1.3b: Python introspection -> KTF builder.

设计 (vs LLMTreeBuilder Phase 4.1):
  LLMTreeBuilder:
    - 输入: 概念名 list
    - 用 LLM 生成节点 (signatures, examples, pitfalls)
    - cost: LLM API
  PythonAPIBuilder (本文件, 新):
    - 输入: API 名 list (e.g. ["pl.scan_csv", "Expr.cum_sum", "Expr.str.contains"])
    - 用 Python introspection 提取 signatures + docstrings (0 cost)
    - 可选: LLM 补全 worked_examples + pitfalls (减少 LLM cost)

Pipeline:
  for each api_name (e.g. "DataFrame.pivot"):
    1. Resolve to actual Python object (introspection)
    2. Extract signature via inspect.signature
    3. Extract docstring via inspect.getdoc
    4. Parse docstring sections (Parameters / Examples / Returns / See Also)
    5. (可选) 调 LLM 补全 worked_examples / pitfalls
    6. Build KnowledgeNode
    7. 如有 storage, 立即保存

支持的 API name 格式:
  - "pl.scan_csv" / "polars.scan_csv": module-level function
  - "DataFrame.pivot" / "pl.DataFrame.pivot": class method
  - "Expr.cum_sum": Expr method
  - "Expr.str.contains" / "ExprStringNameSpace.contains": namespace method
  - "pl.Categorical": class
  - "pl.col": special (returns Expr, signature unavailable)

PROTO 关联:
  PROTO-7.1 (grep 复用): 复用 LLMTreeBuilder 的 BuilderConfig 和验证
  PROTO-7.4 (实测校准): 与手工 32 节点 KTF 对照
  PROTO-7.6 (不基于"应该 work"): 每个 API 实测可解析
  PROTO-7.16 (借理念不依赖工具): 0 外部依赖 (只用 inspect)
"""

import importlib
import inspect
import logging
import re
import time
from typing import Any, Callable, Optional

from .core import KnowledgeNode, WorkedExample
from .storage import KnowledgeStorage
from .builders import (
    BuilderConfig,
    LLMCallable,
    TreeBuilder,
)


class BuilderError(Exception):
    """PythonAPIBuilder 内部错误."""

logger = logging.getLogger(__name__)


# ============================================================================
# API name parsing (resolve string -> Python object)
# ============================================================================

def parse_api_name(api_name: str, module_aliases: Optional[dict] = None) -> tuple:
    """
    解析 API name string 为 (object_path, qualified_name).
    
    Args:
        api_name: "pl.scan_csv", "DataFrame.pivot", "Expr.str.contains" 等
        module_aliases: e.g. {"pl": "polars"}
    
    Returns:
        (object_path_parts, normalized_qualified_name)
        e.g. ("pl.scan_csv", ["polars", "scan_csv"], "polars.scan_csv")
    
    Examples:
        >>> parse_api_name("pl.scan_csv")
        (["polars", "scan_csv"], "polars.scan_csv")
        >>> parse_api_name("DataFrame.pivot")
        (["polars", "DataFrame", "pivot"], "polars.DataFrame.pivot")
        >>> parse_api_name("Expr.str.contains")
        (["polars", "Expr", "str", "contains"], "polars.Expr.str.contains")
    """
    aliases = module_aliases or {"pl": "polars"}
    
    parts = api_name.split(".")
    if not parts:
        raise ValueError(f"Empty api_name: {api_name!r}")
    
    # Replace alias
    if parts[0] in aliases:
        parts[0] = aliases[parts[0]]
    elif parts[0] not in ("polars",) and parts[0] in ("DataFrame", "LazyFrame", "Series", "Expr"):
        # "DataFrame.pivot" -> assume polars prefix
        parts = ["polars"] + parts
    
    qualified = ".".join(parts)
    return (parts, qualified)


def resolve_api_object(parts: list, polars_module: Optional[Any] = None) -> Any:
    """
    解析 parts -> Python object (function / method / class).
    
    Args:
        parts: e.g. ["polars", "scan_csv"] or ["polars", "Expr", "str", "contains"]
        polars_module: 已 import 的 polars 模块 (避免重复 import)
    
    Returns:
        Python object (callable)
    
    Raises:
        AttributeError if path not found
    """
    if polars_module is None:
        polars_module = importlib.import_module(parts[0])
    
    obj = polars_module
    
    # Special case: namespace properties (str, dt, list, struct, cat)
    # 需要 instance 访问获得真正的 namespace 类
    NAMESPACE_NAMES = {"str", "dt", "list", "struct", "cat", "arr",
                        "name", "bin", "meta"}
    
    for i, part in enumerate(parts[1:], 1):
        # Check if this is a namespace property
        if (part in NAMESPACE_NAMES and i < len(parts) - 1
                and hasattr(obj, part)
                and isinstance(getattr(type(obj) if not isinstance(obj, type) else obj, part, None), property)):
            # Need to access via instance to get namespace class
            # For Expr namespaces, create dummy Expr first
            if obj is polars_module.Expr:
                dummy = polars_module.col("__tmp__")
                ns_obj = getattr(dummy, part)
                obj = type(ns_obj)
                continue
            elif obj is polars_module.Series:
                # Similar for Series namespaces
                # 简化: 当前不需要
                pass
        
        obj = getattr(obj, part)
    
    return obj


# ============================================================================
# Docstring parsing
# ============================================================================

# numpy-style section headers (Polars 用 numpy/pandas 风格)
NUMPY_SECTIONS = {
    "Parameters", "Returns", "Yields", "Receives", "Raises",
    "Warns", "Other Parameters", "Attributes", "Methods",
    "See Also", "Notes", "Warnings", "References", "Examples",
}


def parse_numpy_docstring(doc: str) -> dict:
    """
    解析 numpy 风格 docstring 为 sections.
    
    输入:
        Lazily read from a CSV file.
        
        Parameters
        ----------
        source : str
            Path to file.
        
        Returns
        -------
        LazyFrame
        
        Examples
        --------
        >>> pl.scan_csv("data.csv").collect()
    
    输出:
        {
            "Description": "Lazily read from a CSV file.",
            "Parameters": "source : str\\n    Path to file.",
            "Returns": "LazyFrame",
            "Examples": ">>> pl.scan_csv(\"data.csv\").collect()",
        }
    """
    if not doc:
        return {"Description": ""}
    
    lines = doc.split("\n")
    sections = {"Description": []}
    current = "Description"
    
    i = 0
    while i < len(lines):
        line = lines[i]
        # Detect section header: section name on one line, dashes on next
        if (i + 1 < len(lines)
                and line.strip() in NUMPY_SECTIONS
                and lines[i + 1].strip().startswith("-")
                and set(lines[i + 1].strip()) == {"-"}):
            current = line.strip()
            sections[current] = []
            i += 2  # skip header + dashes
            continue
        sections[current].append(line)
        i += 1
    
    # Strip and join
    return {k: "\n".join(v).strip() for k, v in sections.items() if v}


def extract_doctest_examples(doc: str) -> list:
    """
    从 docstring 中提取 >>> doctest 示例.
    
    返回 list of (code_lines, expected_output).
    用作 worked_examples 的来源 (worked_examples 不需要 LLM 生成).
    """
    if not doc:
        return []
    
    examples = []
    lines = doc.split("\n")
    
    current_code = []
    current_output = []
    in_example = False
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(">>>"):
            if in_example and current_code:
                # Save previous
                examples.append({
                    "code": "\n".join(current_code),
                    "output": "\n".join(current_output).strip(),
                })
            current_code = [stripped[3:].strip()]
            current_output = []
            in_example = True
        elif stripped.startswith("...") and in_example:
            current_code.append(stripped[3:].strip())
        elif in_example and stripped:
            current_output.append(stripped)
        elif not stripped and in_example and current_code:
            # Blank line ends example
            examples.append({
                "code": "\n".join(current_code),
                "output": "\n".join(current_output).strip(),
            })
            current_code = []
            current_output = []
            in_example = False
    
    # Last example
    if in_example and current_code:
        examples.append({
            "code": "\n".join(current_code),
            "output": "\n".join(current_output).strip(),
        })
    
    return examples


# ============================================================================
# Safe signature extraction
# ============================================================================

def safe_signature(fn: Any) -> str:
    """Robust signature extraction. 处理特殊情况 (e.g. pl.col 返回 Expr)."""
    try:
        return str(inspect.signature(fn))
    except (TypeError, ValueError):
        return "(...)"


# ============================================================================
# PythonAPIBuilder
# ============================================================================

class PythonAPIBuilder(TreeBuilder):
    """
    从 Python introspection 直接构造 KnowledgeNode.
    
    用法:
        >>> from knowledge_tree.python_api_builder import PythonAPIBuilder
        >>> builder = PythonAPIBuilder()
        >>> nodes = builder.build_from_api_list([
        ...     "pl.scan_csv",
        ...     "DataFrame.pivot",
        ...     "Expr.cum_sum",
        ...     "Expr.str.contains",
        ... ])
        >>> # 32 nodes, 0 LLM cost (introspection only)
    
    或者用 LLM 补 worked_examples:
        >>> from knowledge_tree.local_model_clients import make_nemotron_retriever
        >>> nemo = make_nemotron_retriever()
        >>> builder = PythonAPIBuilder(llm_callable=nemo, enrich_with_llm=True)
        >>> nodes = builder.build_from_api_list([...])
    """
    
    def __init__(
        self,
        config: Optional[BuilderConfig] = None,
        llm_callable: Optional[LLMCallable] = None,
        enrich_with_llm: bool = False,
        module_aliases: Optional[dict] = None,
        max_doc_chars: int = 2000,
    ) -> None:
        """
        Args:
            config: BuilderConfig (validation params)
            llm_callable: 可选 LLM, 用于补充 worked_examples / pitfalls
            enrich_with_llm: 是否调 LLM 补充内容 (默认 False, 完全 introspection)
            module_aliases: e.g. {"pl": "polars"}
            max_doc_chars: docstring 截断长度 (避免节点过大)
        """
        self.config = config or BuilderConfig()
        self.llm_callable = llm_callable
        self.enrich_with_llm = enrich_with_llm and (llm_callable is not None)
        self.module_aliases = module_aliases or {"pl": "polars"}
        self.max_doc_chars = max_doc_chars
        
        # 统计
        self.total_built = 0
        self.total_failed = 0
        self.total_llm_calls = 0
    
    @property
    def name(self) -> str:
        return "python_api_builder"
    
    def build_from_api_list(
        self,
        api_names: list,
        storage: Optional[KnowledgeStorage] = None,
        skip_on_failure: bool = True,
    ) -> list:
        """
        从 API name list 构造节点.
        
        Args:
            api_names: list of API name strings
            storage: 可选 storage, 增量保存
            skip_on_failure: 失败时是否 skip (vs raise)
        
        Returns:
            list of KnowledgeNode
        """
        import importlib
        polars_module = importlib.import_module("polars")
        
        nodes = []
        for api_name in api_names:
            try:
                node = self._build_one(api_name, polars_module)
                nodes.append(node)
                self.total_built += 1
                if storage is not None:
                    storage.save_node(node)
            except Exception as e:
                self.total_failed += 1
                msg = f"Failed to build {api_name!r}: {e}"
                if skip_on_failure:
                    logger.warning(msg)
                    continue
                else:
                    raise BuilderError(msg) from e
        
        return nodes
    
    def _build_one(self, api_name: str, polars_module: Any) -> KnowledgeNode:
        """从 API name 构造一个 KnowledgeNode."""
        # 1. Resolve to Python object
        parts, qualified = parse_api_name(api_name, self.module_aliases)
        obj = resolve_api_object(parts, polars_module)
        
        # 2. Extract signature
        signature_str = safe_signature(obj)
        
        # 3. Extract docstring
        doc = inspect.getdoc(obj) or ""
        if len(doc) > self.max_doc_chars:
            doc = doc[:self.max_doc_chars] + "\n... (truncated)"
        
        # 4. Parse sections
        sections = parse_numpy_docstring(doc)
        
        # 5. Build node fields
        node_id = self._make_node_id(qualified)
        title = self._make_title(qualified, signature_str)
        definition = sections.get("Description", "") or f"API: {qualified}"
        
        # key_facts: signature + parameters + returns
        key_facts = []
        key_facts.append(f"Signature: {qualified}{signature_str}")
        if "Parameters" in sections:
            params = sections["Parameters"][:500]
            key_facts.append(f"Parameters:\n{params}")
        if "Returns" in sections:
            ret = sections["Returns"][:200]
            key_facts.append(f"Returns: {ret}")
        if "Raises" in sections:
            key_facts.append(f"Raises: {sections['Raises'][:200]}")
        
        # worked_examples: from doctest
        worked_examples = []
        doctests = extract_doctest_examples(doc)
        for dt in doctests[:3]:  # max 3 examples
            code = dt["code"][:500]
            if not code.strip():
                continue
            worked_examples.append(WorkedExample(
                problem=f"Example usage of {qualified}",
                solution_steps=["Apply the API as shown in the docstring."],
                final_answer=code,
                key_insight=f"Standard usage pattern for {qualified}.",
            ))
        
        # 如果没有 doctest 示例, 加一个占位 (满足 min_examples)
        if not worked_examples:
            worked_examples.append(WorkedExample(
                problem=f"Use {qualified}",
                solution_steps=[f"Call {qualified}{signature_str}"],
                final_answer=f"# See signature: {qualified}{signature_str[:100]}",
                key_insight=f"Refer to signature for parameter details.",
            ))
        
        # common_pitfalls: 默认空, 留给 LLM 增强 (Phase 4.2 默认不用 LLM)
        common_pitfalls = []
        # 自动检测: 如果是 1.0+ 重命名 API (如 cum_sum), 加 pitfall
        if "cum_" in qualified.lower():
            common_pitfalls.append(
                f"DO NOT use the Polars 0.x name (without underscore). "
                f"Use {qualified.split('.')[-1]} (Polars 1.0+ snake_case)."
            )
        if "unpivot" in qualified.lower():
            common_pitfalls.append(
                "DO NOT use .melt() — that is Polars 0.x. Use .unpivot()."
            )
        
        # 6. LLM enrichment (可选)
        if self.enrich_with_llm and self.llm_callable is not None:
            try:
                enrichment = self._enrich_with_llm(qualified, signature_str, doc)
                if enrichment.get("pitfalls"):
                    common_pitfalls.extend(enrichment["pitfalls"])
                if enrichment.get("examples") and len(worked_examples) < 2:
                    for ex_dict in enrichment["examples"][:1]:
                        try:
                            worked_examples.append(WorkedExample(
                                problem=ex_dict.get("problem", f"Use {qualified}"),
                                solution_steps=ex_dict.get("solution_steps", ["Apply API"]),
                                final_answer=ex_dict.get("final_answer", "# code"),
                                key_insight=ex_dict.get("key_insight", ""),
                            ))
                        except (ValueError, KeyError):
                            continue
                self.total_llm_calls += 1
            except Exception as e:
                logger.warning("LLM enrichment failed for %s: %s", qualified, e)
        
        # 7. Build node
        # 不强制 min_examples / min_facts (BuilderConfig 默认 ≥ 1)
        node = KnowledgeNode(
            id=node_id,
            title=title,
            definition=definition,
            key_facts=key_facts,
            worked_examples=worked_examples,
            common_pitfalls=common_pitfalls,
            parent_id=None,  # PythonAPIBuilder 不强制层级, 用户后续可设
            children_ids=[],
            related_concepts=[],
            domain_metadata={
                "builder": self.name,
                "api_name": api_name,
                "qualified_name": qualified,
                "polars_version": getattr(polars_module, "__version__", "unknown"),
                "signature": signature_str,
                "extracted_at": time.strftime("%Y-%m-%d"),
            },
            source="python_introspection",
        )
        return node
    
    def _make_node_id(self, qualified: str) -> str:
        """从 qualified name 生成 node_id.
        e.g. "polars.scan_csv" -> "polars_scan_csv"
             "polars.DataFrame.pivot" -> "polars_df_pivot"
             "polars.Expr.str.contains" -> "polars_str_contains"
        """
        # Drop "polars." prefix
        if qualified.startswith("polars."):
            tail = qualified[len("polars."):]
        else:
            tail = qualified
        # Normalize class names
        tail = tail.replace("DataFrame.", "df_")
        tail = tail.replace("LazyFrame.", "lf_")
        tail = tail.replace("Expr.", "")
        tail = tail.replace("Series.", "series_")
        tail = tail.replace(".", "_")
        return f"polars_{tail.lower()}"
    
    def _make_title(self, qualified: str, signature: str) -> str:
        """Title: short qualified name."""
        # Show short form: 'pl.scan_csv()' not 'polars.scan_csv(...)'
        short = qualified.replace("polars.", "pl.")
        return f"{short}{signature[:50]}{'...' if len(signature) > 50 else ''}"
    
    def _enrich_with_llm(self, qualified: str, signature: str, doc: str) -> dict:
        """用 LLM 补充 pitfalls + extra examples."""
        prompt = f"""Given the Polars 1.0+ API documentation below, generate:
1. 1-2 common PITFALLS (e.g., "DO NOT use pandas-style X")
2. 1 WORKED EXAMPLE not present in the docstring

API: {qualified}
Signature: {signature}
Docstring:
{doc[:800]}

Output strictly as JSON:
{{"pitfalls": ["...", "..."], "examples": [{{"problem": "...", "final_answer": "..."}}]}}"""
        
        response = self.llm_callable(prompt)
        # Parse JSON response (robust)
        import json
        # Strip ```json fences
        text = response.strip()
        if text.startswith("```"):
            text = re.sub(r'^```(?:json)?\n?', '', text)
            text = re.sub(r'\n?```$', '', text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}
    
    # ====================================================================
    # TreeBuilder protocol (legacy LLMTreeBuilder interface)
    # ====================================================================
    
    def build_from_concepts(
        self,
        concept_names: list,
        parent_concept: Optional[str] = None,
        storage: Optional[KnowledgeStorage] = None,
    ) -> list:
        """
        TreeBuilder 兼容接口: 把 concept_names 当 API names 用.
        """
        return self.build_from_api_list(concept_names, storage=storage)
    
    def get_stats(self) -> dict:
        return {
            "name": self.name,
            "total_built": self.total_built,
            "total_failed": self.total_failed,
            "total_llm_calls": self.total_llm_calls,
        }
