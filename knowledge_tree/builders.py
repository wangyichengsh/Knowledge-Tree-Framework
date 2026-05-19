"""
knowledge_tree/builders.py
==========================

KTF v2 第 3 层 - TreeBuilder 抽象 + LLMTreeBuilder 实现 (Claude API)

框架依赖:
  framework v3.4 T-3.7: worked_examples 是 3x 提升因子, builder 必须生成
  framework v3.4 T-3.8: Domain-Agnostic, builder 接口跨域复用
  framework v3.4 PROTO-7.12 (RAG doc 不作弊): worked_examples 参数不能等于 target
  framework v3.4 PROTO-7.16 (借理念不依赖工具): llm_callable 接口
  framework v3.4 用户决策: builder LLM = Claude API (避免 R1-Distill 自我循环)

设计原则:
  (1) Builder 不绑定特定 LLM
      用 LLMCallable = Callable[[str], str] 接口
      Claude API / 本地 R1-Distill / mock 都能套
      实际 Phase 4.1 用 Claude (用户已决策)
      
  (2) 增量保存 (PROTO-7.4 实测校准)
      每生成 1 个节点立即保存到 storage, 崩溃可恢复
      理由: 500 节点 × 10s/节点 = ~80 分钟, 中途崩溃损失大
      
  (3) 强制结构化输出 + 容错重试
      Claude 用 JSON 输出, 解析失败 retry 1 次
      重试时附加 "previous response was malformed" hint
      仍失败则跳过该节点 (logging error)
      
  (4) PROTO-7.12 防作弊
      可选 target_problems 参数: 已知会用此 builder 输出测试的题目
      builder 检查 worked_example.problem 不含 target 关键参数
      不知道 target 时跳过检查 (一般 corpus 建设)

  (5) 并行支持 (但默认串行)
      Phase 4.1 串行简单, 适合 50-500 节点
      Phase 4.2+ 5000+ 节点时启用 ThreadPoolExecutor (TODO marker)
      不立即实施: API rate limit 复杂处理留 Phase 4.2 一起

文献依据:
  Claude API best practices (Anthropic docs):
    - System prompt for role; User prompt for task
    - JSON mode via explicit instruction + example
    - Max retry 1-2 (避免无限重试浪费成本)
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from knowledge_tree.core import KnowledgeNode, WorkedExample
from knowledge_tree.storage import KnowledgeStorage


logger = logging.getLogger(__name__)


# ============================================================================
# 类型 (与 retrievers 一致)
# ============================================================================

# LLM 调用接口: prompt -> response
LLMCallable = Callable[[str], str]


# ============================================================================
# Builder 配置 (dataclass, 避免参数过多)
# ============================================================================

@dataclass
class BuilderConfig:
    """
    LLMTreeBuilder 配置.

    设计原因 (PROTO-7.6 不基于"应该 work" 假设):
      每个参数都来自 framework 文档或实测约束, 不脑补
    """
    # === 重试与容错 ===
    max_retries: int = 1
    """JSON parse 失败时重试次数. 文献 (Anthropic): 1-2 即可."""

    retry_delay_s: float = 1.0
    """重试前等待 (秒). API rate limit 简单缓解."""

    skip_on_failure: bool = True
    """重试仍失败时是否跳过此节点. False=raise (调试模式)."""

    # === 输出约束 (T-3.7) ===
    min_worked_examples: int = 1
    """每节点至少 N 个 worked_examples (T-3.7 必需)."""

    target_worked_examples: int = 2
    """builder prompt 中要求 LLM 生成的 worked_examples 目标数量."""

    min_key_facts: int = 2
    """每节点至少 N 个 key_facts."""

    target_key_facts: int = 4
    """prompt 中要求 LLM 生成的 key_facts 目标数量."""

    # === PROTO-7.12 防作弊 ===
    target_problems: Optional[list[str]] = None
    """已知 target 题目 (builder 检查 worked_example 不能与之过分相似)."""

    similarity_threshold: float = 0.4
    """
    worked_example.problem 与 target_problems 3-gram word Jaccard 相似度阈值,
    超过判作弊. 默认 0.4 (PROTO-7.4 实测校准, demo_builders.py 实测 0.625 触发).
    """

    # === Related concepts 约束 (Phase 4.1 fix, PROTO-7.4 实测发现) ===
    available_concept_ids: Optional[list[str]] = None
    """
    可用 concept ids 白名单. 若提供, related_concepts 字段必须从此列表中选.
    
    背景 (PROTO-7.4 实测发现):
      Phase 4.1 Week 2 build 前 20 节点实测显示 91% related_concepts 引用了
      hierarchy 中不存在的 concept ids (e.g. Claude 写 'quadratic_equations',
      但 hierarchy 中实际是 'systems_of_equations' / 'polynomial_roots_and_coefficients').
      
    Fix:
      builder prompt 加入 "MUST select related_concepts ONLY from: [...]" 约束
      让 Claude 在已规划的 ids 中选择, 而非自由发挥.
      
    若为 None: builder 不约束 (回退到旧行为, 用于测试 / 不需要约束的场景).
    """
    # === 增量存储 ===
    incremental_save: bool = True
    """每生成 1 个节点立即存储 (默认开)."""

    # === 节点 source 字段值 ===
    source_label: str = "claude_api_builder"

    # === Verbose ===
    verbose: bool = True
    """打印每个节点生成进度."""


# ============================================================================
# Layer 3.0: TreeBuilder ABC
# ============================================================================

class TreeBuilder(ABC):
    """
    建树抽象. 子类实现具体策略.

    职责:
      给定 concept 列表 / 源文档, 生成 KnowledgeNode 集合

    职责边界:
      - NOT: 不持有 retriever (retrievers.py 职责)
      - NOT: 不做检索 (builder 只构造)

    工程实践:
      builder 输出节点 -> storage.save_node
      所有 builder 都接受可选 storage 参数, 实现增量保存
    """

    @abstractmethod
    def build_from_concepts(
        self,
        concept_names: list[str],
        parent_concept: Optional[str] = None,
        storage: Optional[KnowledgeStorage] = None,
    ) -> list[KnowledgeNode]:
        """
        给定概念名清单, 生成 KnowledgeNode 集合.

        Args:
            concept_names: 概念名 list (e.g. ["binomial coefficient", ...])
            parent_concept: 可选父概念 (用于设 parent_id, 但 builder 不强制建关系)
            storage: 可选 storage. 提供则增量保存; 否则只返回内存对象

        Returns:
            list of KnowledgeNode. 顺序与 concept_names 对齐.
            个别失败节点可能缺失 (skip_on_failure=True 时).
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """builder 标识 (用于实验记录)."""


# ============================================================================
# Layer 3.1: LLMTreeBuilder - Phase 4.1 主用
# ============================================================================

class LLMTreeBuilder(TreeBuilder):
    """
    通过 LLM 调用生成 KnowledgeNode.

    Pipeline:
      for each concept_name:
        1. 构造 prompt (含 concept + 输出格式要求 + 防作弊提示)
        2. 调 llm_callable
        3. 解析 JSON 响应
        4. 验证 (worked_examples ≥ min, key_facts ≥ min, 防作弊)
        5. 构造 KnowledgeNode
        6. 如有 storage, 立即保存
      return list

    用法:
        from anthropic import Anthropic
        client = Anthropic(api_key=...)

        def claude_callable(prompt: str) -> str:
            resp = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text

        builder = LLMTreeBuilder(claude_callable)
        nodes = builder.build_from_concepts(
            ["binomial coefficient", "lattice path"],
            parent_concept="combinatorics",
            storage=storage,
        )
    """

    DEFAULT_PROMPT_TEMPLATE = """You are an expert knowledge engineer building a domain-agnostic knowledge tree for mathematical reasoning.

Your task: Generate a structured knowledge node for the concept "{concept_name}".{parent_context}

Output JSON with this EXACT schema:
{{
  "title": "concise concept name (string, 3-8 words)",
  "definition": "1-2 sentence definition (string, 50-200 chars)",
  "key_facts": [
    "fact 1: formula or theorem in mathematical notation",
    "fact 2: ..."
  ],
  "worked_examples": [
    {{
      "problem": "example problem statement (use DIFFERENT parameters than typical exam questions)",
      "solution_steps": ["step 1", "step 2", "..."],
      "final_answer": "the answer",
      "key_insight": "the trick/pattern this example demonstrates (1-2 sentences)"
    }}
  ],
  "common_pitfalls": [
    "pitfall 1: common mistake when applying this concept",
    "pitfall 2: ..."
  ],
  "related_concepts": ["concept_id_1", "concept_id_2"]
}}

Requirements:
- worked_examples: provide AT LEAST {target_worked_examples} examples with COMPLETE solution steps
- key_facts: AT LEAST {target_key_facts} facts (formulas, theorems, invariants)
- common_pitfalls: 1-3 pitfalls
- worked_examples MUST use different specific parameters/values than canonical textbook problems (avoid trivial reproduction){anti_cheat_constraint}
- Use mathematical notation in plain text (e.g. "C(n, k)" not LaTeX){related_concepts_constraint}

Output ONLY the JSON object. No preamble, no markdown code block."""

    RELATED_CONCEPTS_CONSTRAINT_TEMPLATE = """
- IMPORTANT: related_concepts MUST be selected ONLY from this list of available concept IDs:
{available_ids_listing}
  Select 3-6 most relevant IDs from the list above. Do NOT invent new IDs or use variants.
  If none from the list are truly related, return an empty list [] rather than fabricating."""


    def __init__(
        self,
        llm_callable: LLMCallable,
        config: Optional[BuilderConfig] = None,
    ) -> None:
        self.llm_callable = llm_callable
        self.config = config or BuilderConfig()

    @property
    def name(self) -> str:
        return "llm_tree_builder"

    # === Public 入口 ===

    def build_from_concepts(
        self,
        concept_names: list[str],
        parent_concept: Optional[str] = None,
        storage: Optional[KnowledgeStorage] = None,
    ) -> list[KnowledgeNode]:
        """主入口. 详见 ABC.
        
        v2 修改 (Phase 4.1 Week 2 实测发现):
          如果 storage 中已有同 id 节点, 跳过 LLM 调用直接返回 storage 中的节点.
          这避免:
            - 浪费 cost (重新生成已有节点)
            - children_ids 被覆盖 (重新构造的节点 children_ids=[], 
              会破坏 build_tree_with_hierarchy 的双向关系合并)
            - LoRA fine-tune 等下游用户依赖的节点 id 稳定性
        """
        nodes: list[KnowledgeNode] = []
        start_time = time.time()

        for idx, concept_name in enumerate(concept_names, 1):
            # === v2 skip 机制 ===
            if storage is not None:
                node_id = self._make_node_id(concept_name)
                existing = None
                if hasattr(storage, "get_node"):
                    try:
                        existing = storage.get_node(node_id)
                    except KeyError:
                        existing = None
                if existing is not None:
                    if self.config.verbose:
                        logger.info(
                            "LLMTreeBuilder: [%d/%d] %r 已存在 (id=%r), 跳过 LLM 调用",
                            idx, len(concept_names), concept_name, node_id,
                        )
                    nodes.append(existing)
                    continue

            node = self._build_single_node(concept_name, parent_concept)
            if node is None:
                # 失败, 跳过 (skip_on_failure 已处理)
                continue
            nodes.append(node)

            # 增量保存
            if storage is not None and self.config.incremental_save:
                try:
                    storage.save_node(node)
                    if hasattr(storage, "flush"):
                        storage.flush()
                except Exception as e:
                    logger.error(
                        "LLMTreeBuilder: 节点 %r 保存失败 %s (内存中节点仍保留)",
                        node.id, e,
                    )

            if self.config.verbose:
                elapsed = time.time() - start_time
                avg_t = elapsed / idx
                eta = avg_t * (len(concept_names) - idx)
                logger.info(
                    "LLMTreeBuilder: [%d/%d] %r -> id=%r, t=%.1fs, ETA=%.0fmin",
                    idx, len(concept_names), concept_name, node.id,
                    avg_t, eta / 60,
                )

        # 最终 flush (即使非增量也保存一次)
        if storage is not None and hasattr(storage, "flush"):
            try:
                storage.flush()
            except Exception as e:
                logger.error("LLMTreeBuilder: 最终 flush 失败 %s", e)

        logger.info(
            "LLMTreeBuilder: 完成 %d/%d 节点, 总耗时 %.1f min",
            len(nodes), len(concept_names), (time.time() - start_time) / 60,
        )
        return nodes

    # === 单节点生成 ===

    def _build_single_node(
        self,
        concept_name: str,
        parent_concept: Optional[str],
    ) -> Optional[KnowledgeNode]:
        """
        生成单个节点. 包含重试逻辑.

        Returns:
            KnowledgeNode 或 None (失败且 skip_on_failure=True)
        """
        prompt = self._build_prompt(concept_name, parent_concept)
        previous_error: Optional[str] = None

        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._call_llm(prompt, previous_error)
            except Exception as e:
                logger.error(
                    "LLMTreeBuilder: %r LLM 调用失败 (attempt %d/%d): %s",
                    concept_name, attempt + 1, self.config.max_retries + 1, e,
                )
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay_s)
                    previous_error = f"Previous attempt failed with exception: {e}"
                    continue
                # 最终失败
                if self.config.skip_on_failure:
                    return None
                raise

            # 解析 JSON
            parse_result = self._parse_response(response)
            if parse_result is None:
                # JSON 解析失败
                if attempt < self.config.max_retries:
                    logger.warning(
                        "LLMTreeBuilder: %r JSON 解析失败 (attempt %d), 重试",
                        concept_name, attempt + 1,
                    )
                    time.sleep(self.config.retry_delay_s)
                    previous_error = (
                        "Your previous response could not be parsed as valid JSON. "
                        "Please ensure you output ONLY a valid JSON object, "
                        "no markdown, no preamble."
                    )
                    continue
                if self.config.skip_on_failure:
                    return None
                raise ValueError(f"无法解析 LLM 响应 for concept {concept_name!r}")

            # 验证 + 构造 node
            try:
                node = self._construct_node(
                    concept_name, parent_concept, parse_result,
                )
                # PROTO-7.12 防作弊
                if not self._anti_cheat_check(node):
                    logger.warning(
                        "LLMTreeBuilder: %r 防作弊检查失败 (attempt %d)",
                        concept_name, attempt + 1,
                    )
                    if attempt < self.config.max_retries:
                        previous_error = (
                            "Your worked_examples were too similar to target problems. "
                            "Please use significantly different parameters or scenarios."
                        )
                        time.sleep(self.config.retry_delay_s)
                        continue
                    if self.config.skip_on_failure:
                        return None
                    raise ValueError(
                        f"防作弊检查失败 for concept {concept_name!r}"
                    )
                return node
            except (ValueError, KeyError) as e:
                logger.warning(
                    "LLMTreeBuilder: %r 构造节点失败 (attempt %d): %s",
                    concept_name, attempt + 1, e,
                )
                if attempt < self.config.max_retries:
                    previous_error = (
                        f"Your previous response had a validation error: {e}. "
                        f"Please ensure all required fields are present and valid."
                    )
                    time.sleep(self.config.retry_delay_s)
                    continue
                if self.config.skip_on_failure:
                    return None
                raise

        # 不应到达
        return None

    # === Prompt 构造 ===

    def _build_prompt(
        self,
        concept_name: str,
        parent_concept: Optional[str],
    ) -> str:
        """构造 builder prompt."""
        parent_context = ""
        if parent_concept:
            parent_context = (
                f"\n\nThis concept is a sub-topic of: \"{parent_concept}\". "
                f"Frame the definition in relation to this parent context."
            )

        anti_cheat_constraint = ""
        if self.config.target_problems:
            # 不展示完整 target_problems (避免 LLM 直接照搬反作弊),
            # 仅提示概念上下文
            anti_cheat_constraint = (
                "\n- IMPORTANT: This knowledge will be used for general reasoning, "
                "not memorization. Use HYPOTHETICAL or GENERAL parameters in examples."
            )

        # === NEW (Phase 4.1 fix): related_concepts 白名单约束 ===
        related_concepts_constraint = ""
        if self.config.available_concept_ids:
            # 排除当前正在 build 的 concept 自身 (避免自引用)
            current_id = self._make_node_id(concept_name)
            available = [
                cid for cid in self.config.available_concept_ids
                if cid != current_id
            ]
            # 按字母排序便于稳定 + 折成多列展示节省 prompt tokens
            available_sorted = sorted(available)
            # 简单展示: 每行 4 个, 减少 prompt 长度
            lines = []
            for i in range(0, len(available_sorted), 4):
                row = available_sorted[i : i + 4]
                lines.append("  " + ", ".join(row))
            available_ids_listing = "\n".join(lines)
            related_concepts_constraint = self.RELATED_CONCEPTS_CONSTRAINT_TEMPLATE.format(
                available_ids_listing=available_ids_listing,
            )

        return self.DEFAULT_PROMPT_TEMPLATE.format(
            concept_name=concept_name,
            parent_context=parent_context,
            target_worked_examples=self.config.target_worked_examples,
            target_key_facts=self.config.target_key_facts,
            anti_cheat_constraint=anti_cheat_constraint,
            related_concepts_constraint=related_concepts_constraint,
        )

    # === LLM 调用 ===

    def _call_llm(
        self,
        prompt: str,
        previous_error: Optional[str] = None,
    ) -> str:
        """调用 llm_callable. previous_error 用于重试 hint."""
        if previous_error:
            prompt = (
                f"[Retry hint]: {previous_error}\n\n"
                f"---\n\n"
                f"{prompt}"
            )
        return self.llm_callable(prompt)

    # === 响应解析 ===

    def _parse_response(self, response: str) -> Optional[dict[str, Any]]:
        """
        解析 LLM JSON 响应. 容错: 失败返回 None.

        与 LLMRetriever._parse_llm_response 类似, 但解析为完整 dict (不只 ids).
        """
        if not response:
            return None

        cleaned = response.strip()
        # 去除 markdown code block
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # 尝试找第一个 {...} block (含嵌套)
            # 用 brace 计数找完整 JSON
            start = cleaned.find("{")
            if start < 0:
                return None
            depth = 0
            for i in range(start, len(cleaned)):
                ch = cleaned[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(cleaned[start : i + 1])
                        except json.JSONDecodeError:
                            return None
            return None

    # === 节点构造 + 验证 ===

    def _construct_node(
        self,
        concept_name: str,
        parent_concept: Optional[str],
        parsed: dict[str, Any],
    ) -> KnowledgeNode:
        """
        从解析 dict 构造 KnowledgeNode. 验证字段完整性.

        Raises:
            ValueError: 字段缺失 / 类型错误 / 数量不足
            KeyError: 必需 key 缺失
        """
        # 必需字段
        title = parsed.get("title")
        if not title:
            raise ValueError("LLM 响应缺 'title'")

        definition = parsed.get("definition")
        if not definition:
            raise ValueError("LLM 响应缺 'definition'")

        # 数量约束
        key_facts = parsed.get("key_facts", [])
        if not isinstance(key_facts, list):
            raise ValueError(f"key_facts 应为 list, 实际 {type(key_facts).__name__}")
        if len(key_facts) < self.config.min_key_facts:
            raise ValueError(
                f"key_facts 数量 {len(key_facts)} < min {self.config.min_key_facts}"
            )

        worked_examples_raw = parsed.get("worked_examples", [])
        if not isinstance(worked_examples_raw, list):
            raise ValueError(
                f"worked_examples 应为 list, 实际 {type(worked_examples_raw).__name__}"
            )
        if len(worked_examples_raw) < self.config.min_worked_examples:
            raise ValueError(
                f"worked_examples 数量 {len(worked_examples_raw)} "
                f"< min {self.config.min_worked_examples} (T-3.7 关键)"
            )

        # 构造 WorkedExample (会调 __post_init__ 验证)
        worked_examples: list[WorkedExample] = []
        for i, ex_dict in enumerate(worked_examples_raw):
            if not isinstance(ex_dict, dict):
                raise ValueError(f"worked_examples[{i}] 应为 dict")
            try:
                ex = WorkedExample(
                    problem=ex_dict["problem"],
                    solution_steps=list(ex_dict["solution_steps"]),
                    final_answer=str(ex_dict["final_answer"]),
                    key_insight=ex_dict.get("key_insight", ""),
                )
            except (KeyError, ValueError) as e:
                raise ValueError(f"worked_examples[{i}] 构造失败: {e}")
            worked_examples.append(ex)

        common_pitfalls = parsed.get("common_pitfalls", [])
        if not isinstance(common_pitfalls, list):
            common_pitfalls = []

        related_concepts_raw = parsed.get("related_concepts", [])
        if not isinstance(related_concepts_raw, list):
            related_concepts_raw = []
        related_concepts_strs = [str(r) for r in related_concepts_raw]

        # === NEW (Phase 4.1 fix): related_concepts 白名单过滤 ===
        # 即使 prompt 约束了, LLM 仍可能不严格遵守, 这里做最后过滤.
        # 不抛错 (避免触发 retry), 只 log + 过滤
        node_id_for_filter = self._make_node_id(concept_name)
        rejected_refs: list[str] = []
        if self.config.available_concept_ids:
            available_set = set(self.config.available_concept_ids)
            kept = []
            for r in related_concepts_strs:
                if r in available_set and r != node_id_for_filter:  # 排除自引用
                    kept.append(r)
                else:
                    rejected_refs.append(r)
            related_concepts_strs = kept
            if rejected_refs:
                logger.warning(
                    "LLMTreeBuilder: concept %r 的 related_concepts 中 %d 个引用"
                    "不在白名单 (LLM 没严格遵守约束), 已过滤: %s",
                    concept_name, len(rejected_refs), rejected_refs[:5],
                )

        # 生成 id (从 concept_name)
        node_id = self._make_node_id(concept_name)

        # parent_id (如有)
        parent_id = (
            self._make_node_id(parent_concept) if parent_concept else None
        )

        return KnowledgeNode(
            id=node_id,
            title=str(title),
            definition=str(definition),
            key_facts=[str(f) for f in key_facts],
            worked_examples=worked_examples,
            common_pitfalls=[str(p) for p in common_pitfalls],
            parent_id=parent_id,
            children_ids=[],  # builder 不自动建关系, 留给上层 orchestrator
            related_concepts=related_concepts_strs,
            confidence=1.0,
            source=self.config.source_label,
            last_verified=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            domain_metadata={
                "concept_name_raw": concept_name,
                **(
                    {"rejected_related_refs": rejected_refs}
                    if rejected_refs else {}
                ),
            },
        )

    def _make_node_id(self, concept_name: str) -> str:
        """
        从概念名生成 node id.

        规则:
          - lowercase
          - 空白 + 标点 -> '_'
          - 仅保留 [a-z0-9_]
          - 多个 '_' 合并
          - 头尾去 '_'
        """
        # lowercase + 替换非字母数字为 '_'
        s = re.sub(r"[^a-z0-9_]", "_", concept_name.lower())
        # 合并多个 '_'
        s = re.sub(r"_+", "_", s)
        # 去首尾 '_'
        return s.strip("_")

    # === 防作弊检查 (PROTO-7.12) ===

    def _anti_cheat_check(self, node: KnowledgeNode) -> bool:
        """
        检查 worked_examples 是否与 target_problems 过于相似.

        Returns:
            True = 通过检查 (没有作弊)
            False = 失败 (worked_example 与 target 太相似)

        策略 (Phase 4.1 baseline):
          - 3-gram word Jaccard 相似度 (词级 n-gram, 区分度好)
          - 阈值 self.config.similarity_threshold (默认 0.85)
          - 如果 config.target_problems 为 None, 默认通过

        PROTO-7.4 实测发现 (demo_builders.py):
          原版字符级 Jaccard 在长文本上区分度低 (a-z 全集覆盖)
          升级为词级 3-gram Jaccard, 短文本/长文本都有合理区分度

        Phase 4.2 升级路径 (如召回 spurious 节点导致 test set 作弊嫌疑):
          - LLM judge (调 Claude 判断 example 是否"过分相似" target)
          - 语义相似度 (sentence embedding, 但破坏 vectorless 约束)
          - 关键参数提取 + 精确匹配 (e.g. "5x4 grid" 在两边都出现 -> 作弊)
        """
        if not self.config.target_problems:
            return True

        def word_ngrams(text: str, n: int = 3) -> set[str]:
            """3-gram words. 短文本 (< 3 词) 退化为词集."""
            words = text.lower().split()
            if len(words) < n:
                return set(words)
            return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}

        for ex in node.worked_examples:
            ex_ngrams = word_ngrams(ex.problem)
            for target in self.config.target_problems:
                target_ngrams = word_ngrams(target)
                if not ex_ngrams or not target_ngrams:
                    continue
                # Jaccard: |A ∩ B| / |A ∪ B|
                intersection = ex_ngrams & target_ngrams
                union = ex_ngrams | target_ngrams
                jaccard = len(intersection) / len(union)
                if jaccard >= self.config.similarity_threshold:
                    logger.warning(
                        "PROTO-7.12: worked_example 与 target 过于相似 "
                        "(3-gram jaccard=%.3f >= %.3f)\n  example: %r\n  target:  %r",
                        jaccard, self.config.similarity_threshold,
                        ex.problem[:80], target[:80],
                    )
                    return False
        return True


# ============================================================================
# Layer 3.2 / 3.3 / 3.4: Phase 4.2+ stubs (不实施)
# ============================================================================

class ASTTreeBuilder(TreeBuilder):
    """
    Phase 4.2 stub. 代码域 AST 解析构造节点.

    实施触发条件 (architecture v1.11 第十五·B 章):
      - Phase 4.2 Week 5+ 代码域扩展
      - 输入: GitHub repo / Python package
      - 输出: 每个函数/类是 1 个 KnowledgeNode
              key_facts = signature + docstring + return type
              worked_examples = test cases / docstring examples
              related_concepts = call graph 邻居

    需要的库:
      - ast (Python builtin)
      - libcst (语法树编辑, 可选)
      - cAST (架构第十五章引用的工具)

    设计预留:
      接受 module_path / repo_url 参数
      内部解析 -> 构造 KnowledgeNode
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "ASTTreeBuilder 是 Phase 4.2 stub, 当前未实施. "
            "Phase 4.1 用 LLMTreeBuilder."
        )

    def build_from_concepts(
        self,
        concept_names: list[str],
        parent_concept: Optional[str] = None,
        storage: Optional[KnowledgeStorage] = None,
    ) -> list[KnowledgeNode]:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return "ast_tree_builder"


class HybridTreeBuilder(TreeBuilder):
    """
    Phase 4.2 stub. LLM + AST 组合.

    设计意图:
      AST 提取结构 (函数签名/调用图), LLM 生成 worked_examples / explanations
      混合用例: 代码库已知 API 结构, LLM 补充语义层

    实施触发条件:
      Phase 4.2 Week 6+ 代码域 RAG 验证成功后
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "HybridTreeBuilder 是 Phase 4.2 stub, 当前未实施."
        )

    def build_from_concepts(
        self,
        concept_names: list[str],
        parent_concept: Optional[str] = None,
        storage: Optional[KnowledgeStorage] = None,
    ) -> list[KnowledgeNode]:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return "hybrid_tree_builder"


# ============================================================================
# Layer 3.X: PDFSourceLoader stub (Phase 4.2+, 用户提议 OpenDataLoader 集成点)
# ============================================================================

class PDFSourceLoader:
    """
    Phase 4.2+ stub. PDF 解析助手, 把 PDF 转 Markdown 供 LLMTreeBuilder 用.

    设计意图 (Phase 4.1 不实施, 仅留接口):
      pdf -> markdown -> 由 LLM 抽取概念清单 -> LLMTreeBuilder 建树

    候选工具 (用户提议, web 搜索验证):
      - OpenDataLoader-PDF (Apache-2.0, 本地, #1 benchmark 0.907 overall)
        opendataloader.org / github.com/opendataloader-project/opendataloader-pdf
        优点: 本地, 数学公式 + 表格 + LaTeX 友好, 不需 GPU
        与 KTF 哲学一致: vectorless + 本地 + PROTO-7.16 (不依赖 vendor)
      - pymupdf4llm (轻量但 table/heading accuracy 较低)
      - marker (准但需 GPU + 慢)

    实施触发条件 (Phase 4.2+):
      (a) 代码域: 解析 GitHub repo 的 README/docs PDF
      (b) 科学域: 解析论文 PDF (e.g. arxiv)
      (c) 真实流水线: 用户提供 PDF 知识源 (教科书 / 报告)

    当前 (Phase 4.1) 不需要:
      LLMTreeBuilder 输入是概念名清单 (不是 PDF), 由人工或现有数据库提供
      AoPS / Wikipedia 数据是网页 (HTML), 不是 PDF
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "PDFSourceLoader 是 Phase 4.2+ stub, 当前未实施.\n"
            "Phase 4.1 LLMTreeBuilder 接受概念名清单, 不需要 PDF 解析.\n"
            "Phase 4.2+ 启用时建议用 OpenDataLoader-PDF (Apache-2.0, 本地, "
            "数学公式友好). 集成示例:\n"
            "  pip install opendataloader-pdf\n"
            "  import opendataloader_pdf\n"
            "  opendataloader_pdf.convert(\n"
            "    input_path=['paper.pdf'], output_dir='out/', format='markdown')\n"
            "  # 解析后 markdown 由 LLM 抽取概念清单 -> 传给 LLMTreeBuilder"
        )


# ============================================================================
# 工具函数: 给定一组概念, 一次性建树 + 关系 (Phase 4.1 高级用法)
# ============================================================================

def build_tree_with_hierarchy(
    builder: TreeBuilder,
    hierarchy: dict[str, list[str]],
    storage: Optional[KnowledgeStorage] = None,
) -> list[KnowledgeNode]:
    """
    根据预设层级生成节点 + 自动建立 parent-children 关系.

    Args:
        builder: TreeBuilder 实例
        hierarchy: dict[parent_concept, list[child_concept]]
            e.g. {
                "combinatorics": ["binomial coefficient", "lattice path"],
                "geometry": ["triangle", "circle"],
            }
            空 dict 值表示 root 节点 (无 parent)
        storage: 可选 storage

    Returns:
        所有节点 (parent + children), 含 children_ids 双向关系

    用法:
        nodes = build_tree_with_hierarchy(builder, {
            "combinatorics": [],  # root
            "binomial coefficient": [],  # 注: 这里写法暗示 binomial 也是 root
        }, storage=storage)
        
        # 或者:
        nodes = build_tree_with_hierarchy(builder, {
            "combinatorics": ["binomial coefficient", "permutation"],
        }, storage=storage)

    注意:
        当前实现要求所有 parents 显式列在 keys 中.
        builder.build_from_concepts 已在每个节点设 parent_id,
        本函数补充 children_ids 反向 (维护双向一致 - core.py validate 要求).
    """
    # 收集所有概念 (parents + children)
    all_concepts: dict[str, Optional[str]] = {}  # concept -> parent

    for parent, children in hierarchy.items():
        # parent 自身是 root (除非也被某 child 列了)
        if parent not in all_concepts:
            all_concepts[parent] = None
        for child in children:
            all_concepts[child] = parent

    # 分批生成: 先 roots, 再 children (按 parent 分组)
    nodes_by_concept: dict[str, KnowledgeNode] = {}

    # 先生成 roots
    roots = [c for c, p in all_concepts.items() if p is None]
    if roots:
        root_nodes = builder.build_from_concepts(
            roots, parent_concept=None, storage=storage,
        )
        for n, c in zip(root_nodes, roots):
            nodes_by_concept[c] = n

    # 再按 parent 分组生成 children
    children_by_parent: dict[str, list[str]] = {}
    for c, p in all_concepts.items():
        if p is not None:
            children_by_parent.setdefault(p, []).append(c)

    for parent, children in children_by_parent.items():
        child_nodes = builder.build_from_concepts(
            children, parent_concept=parent, storage=storage,
        )
        for n, c in zip(child_nodes, children):
            nodes_by_concept[c] = n

    # 补 parent.children_ids (双向关系)
    # NOTE (Phase 4.1 Week 2 实测发现的 bug):
    #   v1 直接赋值 parent_node.children_ids = child_ids 会覆盖现有 children
    #   场景: 断点续传时, 当前 hierarchy 只含剩余未 build 的概念,
    #         child_ids 只是这次新增的 children, 不包含之前 build 的
    #         若直接赋值会丢失之前的 children_ids 关系
    #   v2 修复: 合并 (union) 现有 children_ids + 新增 child_ids, 保持顺序去重
    for parent, children in hierarchy.items():
        if parent not in nodes_by_concept:
            continue
        parent_node = nodes_by_concept[parent]
        new_child_ids = [
            nodes_by_concept[c].id
            for c in children
            if c in nodes_by_concept
        ]
        # 合并现有 + 新增, 保序去重 (v2 修复)
        existing_child_ids = list(parent_node.children_ids or [])
        merged_child_ids = list(existing_child_ids)
        for cid in new_child_ids:
            if cid not in merged_child_ids:
                merged_child_ids.append(cid)
        parent_node.children_ids = merged_child_ids

        # 重新保存 parent (children_ids 改了)
        if storage is not None:
            try:
                storage.save_node(parent_node)
            except Exception as e:
                logger.error(
                    "build_tree_with_hierarchy: parent %r 更新 children_ids 失败 %s",
                    parent_node.id, e,
                )

    if storage is not None and hasattr(storage, "flush"):
        try:
            storage.flush()
        except Exception as e:
            logger.error("build_tree_with_hierarchy: 最终 flush 失败 %s", e)

    return list(nodes_by_concept.values())
