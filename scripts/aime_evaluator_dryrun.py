#!/usr/bin/env python3
"""
scripts/aime_evaluator_dryrun.py
================================

Phase 4.1 Week 1 - AIME evaluator 兼容性 dry-run.

目的 (C-5 修正点 4, PROTO-7.9 dual validation):
  在 Phase 4.1 Week 4 启动 AIME 30 题 RAG 实验之前, 先 dry-run:
  
  (1) 验证 Tool 1 v4 evaluator 在 AIME 答案格式上是否兼容 (Bug D 候选)
  (2) 测 R1-Distill+LoRA(checkpoint-50) 在 AIME 上的 baseline accuracy
  (3) 识别 ceiling 题 (KTF 增强目标)
  
此脚本是 Week 4 的"前置 sanity check", 阻塞 Week 4 启动如果发现 Bug D.

设计决策记录:
  (1) 使用 HuggingFaceH4/aime_2024 数据集 (30 题, 字段小写)
      其他候选: Maxwell-Jia/AIME_2024 (字段 PascalCase)
                AI-MO/aimo-validation-aime (90 题, 2022-2024)
      理由: H4 字段命名与你 evaluator_check.py jsonl 风格一致
  
  (2) 不动 phase2_mcts.load_data (它不支持 aime), 在本脚本内 inline 加载
      PROTO-7.16: 业务逻辑不依赖具体工具 (不改底层基础设施)
  
  (3) 复用 evaluator_check.py 的模型加载 + generate_one (4-bit + LoRA + 显存清理)
      PROTO-7.4: 实测校准, 不复制实现避免漂移
  
  (4) 双模式: --dry-run-evaluator-only 只测 evaluator (不跑模型, 几秒)
              否则跑完整 baseline (~30 题 × 5min = 2.5h)
      理由: PROTO-7.9 单元测试 + 实数据 dual-validation 分两步走

实测发现 (本脚本编写前 dry-run):
  AIME 答案是 0-999 整数, Tool 1 v4 evaluator 全部兼容:
    '035' vs '35'        -> True (float 比较)
    '35/1' vs '35'       -> True (sympify)
    '420/12' vs '35'     -> True (sympify)
    '\\boxed{35}' vs '35' -> True (extract_boxed)
  
  无 Bug D 候选. evaluator 修复路径不需要 v5.

用法:
  # 仅测 evaluator 兼容性 (~3 秒, 无需 GPU)
  python scripts/aime_evaluator_dryrun.py --dry-run-evaluator-only \\
    --output evaluator_dryrun_compat.jsonl

  # 完整 baseline (~2-3h, 需 GPU + LoRA)
  python scripts/aime_evaluator_dryrun.py \\
    --base-model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \\
    --explorer-lora models/explorer-grpo-sanity/checkpoint-50 \\
    --output aime_baseline_dryrun.jsonl

输出格式 (jsonl, 与 evaluator_check_n100 兼容):
  {
    "question": ...,
    "gt_answer": ...,    # AIME answer 标准化为 str (e.g. "35")
    "pred_answer": ...,
    "is_correct": bool,
    "level": "AIME",     # 占位 (与 MATH 风格一致)
    "type": ...,         # AIME 主题分类 (如有)
    "sample_idx": 0,
    "full_reasoning": ..., # 模型完整输出
    "config": "aime_dryrun",
    "sample_id": int,    # AIME 题号 (e.g. 0-29)
    "token_count": int,
    "time_s": float,
    "truncated": bool,
    "status": str,       # 'complete' / 'inside_truncated' / 'complete_no_answer'
    "aime_id": str,      # 原始 AIME ID (e.g. "2024-I-1")
  }
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Optional

# 路径配置: 让脚本能 import phase2_mcts (用户上传位置)
# Phase 4.1 实际运行时, 用户应在仓库根目录运行, 这里仅 demo
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


logger = logging.getLogger(__name__)


# ============================================================================
# AIME 数据加载 (与 phase2_mcts.load_data 解耦, PROTO-7.16)
# ============================================================================

def load_aime_2024(dataset_name: str = "HuggingFaceH4/aime_2024") -> list[dict]:
    """
    加载 AIME 2024 数据集.

    支持的数据集:
      HuggingFaceH4/aime_2024 (默认, 30 题, 字段小写)
        字段: id, problem, solution, answer, url, year
      Maxwell-Jia/AIME_2024 (30 题, 字段 PascalCase)
        字段: ID, Problem, Solution, Answer

    Returns:
        list of dict with normalized keys:
          {
            "question": str,
            "gt_answer": str,  # AIME 答案统一为 str (e.g. "35")
            "aime_id": str,    # 原始 AIME ID
            "level": "AIME",   # 占位
            "type": "AIME",    # 占位 (AIME 不分类)
            "sample_id": int,  # 顺序号 0-29
          }
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "datasets 库未安装. pip install datasets --break-system-packages"
        )

    logger.info("加载数据集: %s", dataset_name)
    ds = load_dataset(dataset_name, split="train")

    # 探测字段命名风格 (HuggingFaceH4 vs Maxwell-Jia)
    sample = ds[0]
    if "problem" in sample:  # HuggingFaceH4 风格
        problem_key, answer_key, id_key = "problem", "answer", "id"
    elif "Problem" in sample:  # Maxwell-Jia 风格
        problem_key, answer_key, id_key = "Problem", "Answer", "ID"
    else:
        raise ValueError(
            f"未知 AIME 数据集字段: {list(sample.keys())}"
        )

    samples = []
    for i, row in enumerate(ds):
        gt = str(row[answer_key]).strip()
        # AIME 答案应该是整数, 验证
        try:
            int_val = int(gt)
            if not 0 <= int_val <= 999:
                logger.warning(
                    "AIME 题 %d 答案 %r 超出 [0, 999] 范围 (异常)",
                    i, gt,
                )
        except ValueError:
            logger.warning(
                "AIME 题 %d 答案 %r 不是整数 (可能数据异常)",
                i, gt,
            )

        samples.append({
            "question": row[problem_key],
            "gt_answer": gt,
            "aime_id": str(row[id_key]),
            "level": "AIME",
            "type": "AIME",
            "sample_id": i,
        })

    logger.info("加载 %d 题 AIME 2024", len(samples))
    return samples


# ============================================================================
# Evaluator 兼容性测试 (Bug D 检测)
# ============================================================================

# 模拟 R1-Distill 在 AIME 上可能的输出格式
# 每个 case: (description, simulated_text, expected_pred, expected_match_with_35)
AIME_OUTPUT_FORMATS = [
    # 标准
    ("standard_boxed", "...</think>\\boxed{35}", "35", True),
    ("standard_long", "thinking...</think>So the answer is \\boxed{35}.", "35", True),
    # Leading zero (Bug D 候选 1)
    ("leading_zero_3", "</think>\\boxed{035}", "035", True),
    ("leading_zero_long", "</think>The answer is \\boxed{035}.", "035", True),
    # 空格 (Bug D 候选)
    ("spaces_inside_boxed", "</think>\\boxed{ 35 }", "35", True),
    # Fraction (Bug D 候选 2)
    ("fraction_redundant", "</think>\\boxed{35/1}", "35/1", True),
    ("fraction_unsimplified", "</think>\\boxed{\\frac{420}{12}}", "420/12", True),
    # LaTeX text wrap
    ("text_wrap", "</think>\\boxed{\\text{35}}", "35", True),
    # 模型故意写错答 (合法格式)
    ("wrong_answer", "</think>\\boxed{42}", "42", False),
    # 撞顶无 boxed
    ("truncated_no_boxed", "Thinking...", "", False),
    # 无 </think> (R1-Distill 偶尔)
    ("no_think_tag", "\\boxed{35}", "35", True),
    # 多个 boxed (取最后)
    ("multiple_boxed", "</think>First \\boxed{50}, then \\boxed{35}", "35", True),
]


def run_evaluator_dryrun(
    extract_fn,
    match_fn,
    target_gt: str = "35",
) -> list[dict]:
    """
    Dry-run evaluator on simulated AIME output formats.

    检测 Bug D 候选: 任何 expected_match=True 但实际 match=False 的 case
    都是潜在 Bug D.

    Args:
        extract_fn: phase2_mcts.extract_answer_from_text
        match_fn: phase2_mcts.answers_match
        target_gt: 模拟 GT 答案 (默认 "35")

    Returns:
        list of dict, 每个 case 的结果
    """
    results = []
    bug_d_candidates = []

    for label, text, expected_pred, expected_match in AIME_OUTPUT_FORMATS:
        try:
            pred = extract_fn(text)
        except Exception as e:
            results.append({
                "case": label,
                "input_text": text,
                "expected_pred": expected_pred,
                "actual_pred": f"<ERROR: {e}>",
                "expected_match": expected_match,
                "actual_match": False,
                "bug_d_candidate": True,
                "note": f"extract_fn raised: {e}",
            })
            bug_d_candidates.append(label)
            continue

        try:
            match = match_fn(pred, target_gt) if pred else False
        except Exception as e:
            results.append({
                "case": label,
                "input_text": text,
                "expected_pred": expected_pred,
                "actual_pred": pred,
                "expected_match": expected_match,
                "actual_match": False,
                "bug_d_candidate": True,
                "note": f"match_fn raised: {e}",
            })
            bug_d_candidates.append(label)
            continue

        # Bug D 候选: 期望 match 但实际不 match (false negative)
        # 注: false positive (期望不 match 但 match) 也是 bug, 但 AIME 中较罕见
        is_bug = (match != expected_match)

        results.append({
            "case": label,
            "input_text": text,
            "expected_pred": expected_pred,
            "actual_pred": pred,
            "expected_match": expected_match,
            "actual_match": match,
            "bug_d_candidate": is_bug,
            "note": "false_negative" if (expected_match and not match)
                    else ("false_positive" if (not expected_match and match) else ""),
        })
        if is_bug:
            bug_d_candidates.append(label)

    return results, bug_d_candidates


# ============================================================================
# 完整 baseline (需要 GPU + LoRA), 复用 evaluator_check.py 风格
# ============================================================================

def load_model(base_model: str, explorer_lora: Optional[str] = None):
    """
    加载 base + LoRA (4-bit). 复用 evaluator_check.py 实现风格.

    与 evaluator_check.py 一致 (PROTO-7.4 实测校准, 不发明新 config).
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    logger.info("加载 base 模型: %s", base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    if explorer_lora:
        from peft import PeftModel
        if not os.path.isdir(explorer_lora):
            raise FileNotFoundError(f"LoRA 路径不存在: {explorer_lora!r}")
        adapter_config = os.path.join(explorer_lora, "adapter_config.json")
        if not os.path.isfile(adapter_config):
            raise FileNotFoundError(
                f"未找到 adapter_config.json at {adapter_config}\n"
                f"  尝试: --explorer-lora {explorer_lora}/checkpoint-50"
            )
        logger.info("加载 LoRA: %s", explorer_lora)
        model = PeftModel.from_pretrained(model, explorer_lora)

    model.eval()
    return model, tokenizer


def generate_one(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 16384,
    temperature: float = 0.6,
) -> tuple[str, int, str]:
    """
    生成单次 response. 复用 evaluator_check.py 的显存清理 (PROTO-7.3).
    """
    import torch

    messages = [{"role": "user", "content": prompt}]
    inputs_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(inputs_text, return_tensors="pt").to(model.device)
    prompt_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    response_ids = outputs[0, prompt_length:].clone()
    response_length = response_ids.shape[0]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

    has_think_close = "</think>" in response_text
    has_boxed = r"\boxed{" in response_text
    if response_length >= max_new_tokens - 10:
        status = "complete" if has_boxed else "inside_truncated"
    else:
        status = "complete" if has_boxed else "complete_no_answer"

    # PROTO-7.3 显存清理
    del outputs, inputs, response_ids
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return response_text, response_length, status


def build_prompt(question: str) -> str:
    """与 evaluator_check.py / cycle 1 GRPO sanity 训练时一致的 prompt."""
    return (
        f"{question}\n\n"
        f"Please reason step by step, and put your final answer within \\boxed{{}}."
    )


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="HuggingFaceH4/aime_2024",
        help="HuggingFace AIME 数据集 (默认 HuggingFaceH4/aime_2024)",
    )
    parser.add_argument(
        "--dry-run-evaluator-only",
        action="store_true",
        help="仅测 evaluator 兼容性, 不跑模型 (~3 秒, 无 GPU 需求)",
    )
    parser.add_argument("--base-model", default=None,
                        help="(完整 baseline 模式) base 模型")
    parser.add_argument("--explorer-lora", default=None,
                        help="(完整 baseline 模式) LoRA checkpoint 路径")
    parser.add_argument("--n", type=int, default=30,
                        help="题数 (默认 30, AIME 2024 全部)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=789,
                        help="随机 seed (与 Phase 1-3 不同避免重叠)")
    parser.add_argument("--output", required=True,
                        help="输出 jsonl 路径")
    parser.add_argument(
        "--phase2-mcts-path",
        default=None,
        help="phase2_mcts.py 路径 (默认从 sys.path 找)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 加载 evaluator
    if args.phase2_mcts_path:
        sys.path.insert(0, os.path.dirname(args.phase2_mcts_path))
    try:
        from phase2_mcts import extract_answer_from_text, answers_match
    except ImportError as e:
        logger.error(
            "无法导入 phase2_mcts.extract_answer_from_text / answers_match: %s\n"
            "  请确保 phase2_mcts.py 在 sys.path 中, 或用 --phase2-mcts-path 指定",
            e,
        )
        sys.exit(1)

    # ====== 阶段 1: Evaluator dry-run (always done) ======
    print("=" * 78)
    print("Phase 4.1 Week 1 - AIME Evaluator Dry-Run")
    print("=" * 78)
    print()
    print("[阶段 1] Evaluator 兼容性测试 (模拟 12 种 AIME 输出格式)")
    print("-" * 78)

    eval_results, bug_d_candidates = run_evaluator_dryrun(
        extract_answer_from_text, answers_match, target_gt="35",
    )

    print(f"\n{'Case':<25} {'Pred':<22} {'Expected':<10} {'Actual':<10} {'Status'}")
    print("-" * 78)
    for r in eval_results:
        status = "✓"
        if r["bug_d_candidate"]:
            status = "❌ BUG_D" if r["note"] == "false_negative" else "⚠️ "
        pred_display = r["actual_pred"] if len(r["actual_pred"]) <= 20 else r["actual_pred"][:17] + "..."
        print(
            f"{r['case']:<25} {pred_display!r:<22} "
            f"{str(r['expected_match']):<10} {str(r['actual_match']):<10} {status}"
        )

    print(f"\n[Evaluator 兼容性总结]")
    print(f"  测试 cases: {len(eval_results)}")
    print(f"  Bug D 候选: {len(bug_d_candidates)}")
    if bug_d_candidates:
        print(f"  ❌ 需要修复 evaluator (Bug D): {bug_d_candidates}")
        print(f"  Phase 4.1 Week 4 AIME 实验阻塞, 必须先修 evaluator")
    else:
        print(f"  ✅ Tool 1 v4 evaluator 与 AIME 答案格式完全兼容")
        print(f"  无需 evaluator v5 修复. Phase 4.1 Week 4 AIME 实验解除阻塞.")

    # ====== 阶段 2: 加载 AIME 数据集 (always) ======
    print(f"\n[阶段 2] 加载 AIME 数据集: {args.dataset}")
    print("-" * 78)
    try:
        aime_samples = load_aime_2024(args.dataset)
    except Exception as e:
        logger.error("AIME 数据集加载失败: %s", e)
        sys.exit(1)
    aime_samples = aime_samples[: args.n]
    print(f"  加载 {len(aime_samples)} 题")
    print(f"  Sample: {aime_samples[0]['question'][:80]}...")
    print(f"          gt={aime_samples[0]['gt_answer']!r}")

    if args.dry_run_evaluator_only:
        # 仅 evaluator 模式: 不跑模型, 写空 baseline (用于 Phase 4.1 Week 1 完成度记录)
        records = []
        for s in aime_samples:
            records.append({
                **s,
                "pred_answer": None,
                "is_correct": False,
                "full_reasoning": "",
                "config": "aime_dryrun_evaluator_only",
                "token_count": 0,
                "time_s": 0,
                "truncated": False,
                "status": "skipped",
                "sample_idx": 0,
            })
        with open(args.output, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n[Dry-run 模式] 跳过模型加载. 仅 evaluator 兼容性已验证.")
        print(f"输出: {args.output} ({len(records)} 题占位)")
        return

    # ====== 阶段 3: 完整 baseline (需 GPU) ======
    if not args.base_model:
        logger.error("完整 baseline 模式需要 --base-model 参数")
        sys.exit(1)

    print(f"\n[阶段 3] 完整 baseline (R1-Distill+LoRA, vanilla generation)")
    print("-" * 78)
    print(f"  Base: {args.base_model}")
    print(f"  LoRA: {args.explorer_lora}")
    print(f"  Samples: {len(aime_samples)} × 1 sample = {len(aime_samples)} generations")
    print(f"  Estimated time: ~{len(aime_samples) * 5}min (5min/题)")

    # 加载模型
    model, tokenizer = load_model(args.base_model, args.explorer_lora)

    import torch
    records = []
    correct_count = 0
    start_time = time.time()
    by_aime_id = {}

    for q_idx, sample in enumerate(aime_samples):
        torch.manual_seed(args.seed * 1000 + q_idx * 100)
        prompt = build_prompt(sample["question"])

        t0 = time.time()
        response_text, response_length, status = generate_one(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        elapsed = time.time() - t0

        pred = extract_answer_from_text(response_text, question=sample["question"])
        is_correct = answers_match(pred, sample["gt_answer"]) if pred else False

        record = {
            "question": sample["question"],
            "gt_answer": sample["gt_answer"],
            "pred_answer": pred,
            "is_correct": is_correct,
            "level": sample["level"],
            "type": sample["type"],
            "sample_idx": 0,
            "full_reasoning": response_text,
            "config": "aime_dryrun_baseline",
            "sample_id": sample["sample_id"],
            "token_count": response_length,
            "time_s": elapsed,
            "truncated": status == "inside_truncated",
            "status": status,
            "aime_id": sample["aime_id"],
        }
        records.append(record)
        correct_count += int(is_correct)
        by_aime_id[sample["aime_id"]] = is_correct

        # 进度 + 显存监控
        # 注: torch.cuda.max_memory_allocated 仅统计 PyTorch tensor 显存
        # 不含 CUDA context / bnb 4bit kernel buffer / activation 等
        # 真实进程显存占用需用 nvidia-smi (4bit 模型差异可达 2-3x)
        # 见 framework v3.5 PROTO-7.22
        avg_t = (time.time() - start_time) / (q_idx + 1)
        eta = avg_t * (len(aime_samples) - q_idx - 1) / 60
        vram_str = ""
        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            cur_gb = torch.cuda.memory_allocated() / 1e9
            # 标签明确为 "torch_peak" 而非 "peak" 避免误读
            vram_str = f"  torch_peak={peak_gb:.1f}/now={cur_gb:.1f}GB (仅PyTorch tensor部分)"
            torch.cuda.reset_peak_memory_stats()

        marker = "✓" if is_correct else "✗"
        print(
            f"  [{q_idx+1}/{len(aime_samples)}] {marker} {sample['aime_id']}: "
            f"pred={pred!r:15.15}, gt={sample['gt_answer']!r:6}, "
            f"len={response_length}, t={elapsed:.0f}s, ETA={eta:.0f}min{vram_str}"
        )

        # 中途保存 (每 5 题)
        if (q_idx + 1) % 5 == 0:
            with open(args.output, "w") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 最终保存
    with open(args.output, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ====== 阶段 4: 总结 ======
    total_elapsed = time.time() - start_time
    print(f"\n{'='*78}")
    print(f"完成 - {len(records)} 题 in {total_elapsed/60:.1f} min")
    print(f"{'='*78}")
    print(f"\n=== AIME 2024 Baseline (R1-Distill+LoRA) ===")
    print(f"Accuracy: {correct_count}/{len(records)} = {correct_count/len(records)*100:.1f}%")
    print(f"  (文献参考: R1-Distill-14B vanilla AIME 2024 pass@1 ≈ 69.7%)")

    # 状态分布
    status_counts = defaultdict(int)
    for r in records:
        status_counts[r["status"]] += 1
    print(f"\n=== Status 分布 ===")
    for s, c in sorted(status_counts.items()):
        print(f"  {s}: {c}/{len(records)}")

    # Ceiling 题列表 (Phase 4.1 Week 4 RAG 增强目标)
    ceiling = [r for r in records if not r["is_correct"]]
    print(f"\n=== Ceiling 题 ({len(ceiling)} 道, KTF 增强目标) ===")
    for r in ceiling[:10]:  # 只展示前 10
        print(
            f"  {r['aime_id']}: gt={r['gt_answer']!r}, pred={r['pred_answer']!r:15.15}, "
            f"status={r['status']}"
        )
    if len(ceiling) > 10:
        print(f"  ... ({len(ceiling) - 10} more)")

    print(f"\n输出: {args.output}")
    print()
    print("Phase 4.1 Week 4 AIME 实验解除阻塞条件:")
    print(f"  - Evaluator 兼容性: {'✅' if not bug_d_candidates else '❌ 阻塞'}")
    print(f"  - Baseline 已建立: ✅ ({correct_count/len(records)*100:.1f}% accuracy)")
    print(f"  - Ceiling 池建立: ✅ ({len(ceiling)} 题)")


if __name__ == "__main__":
    main()
