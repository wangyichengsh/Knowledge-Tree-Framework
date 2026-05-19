# scripts/day5_step5_generate.py
import json, sys, os
sys.path.insert(0, os.environ.get('PROJECT_ROOT', '.'))

from knowledge_tree.local_model_clients import make_r1_generator

TASK_ID = "django__django-11001"

candidates = json.load(open('day5_candidates.json'))
task = candidates['medium']
retrieved = json.load(open(f"/tmp/swe-bench-day5/{TASK_ID}/retrieved.json"))

# Build prompt
inject_blocks = "\n\n".join([f"### Retrieved Code Reference {r['rank']}\n```python\n{r['inject']}\n```"
                              for r in retrieved])

prompt = f"""You are a senior Python engineer fixing a bug in {task['repo']}.

## Bug Report
{task['problem_statement']}

## Repository Context (top-3 relevant functions, BM25 retrieved)

{inject_blocks}

## Task
Generate a git-style unified diff patch that fixes the bug.
The patch should be applicable to the repository at base_commit {task['base_commit'][:12]}.

## Output Format
Output ONLY the patch (start with `diff --git`), no explanation.

## Patch:
"""

print(f"Prompt size: {len(prompt)} chars (~{len(prompt) // 4} tokens)")

# Generate
r1 = make_r1_generator(max_new_tokens=4096, verbose=True)
print("R1 loaded, generating...")

import time
t0 = time.time()
response = r1(prompt)
gen_time = time.time() - t0
print(f"Generated in {gen_time:.1f}s")
print(f"Response size: {len(response)} chars")

# Extract patch (first diff --git block)
import re
# patch_match = re.search(r'diff --git[\s\S]+?(?=\n```|\Z)', response)

def extract_patch(response):
    """Robust patch extraction. 处理 R1 thinking 中 partial diff."""
    # 找所有 diff --git 块
    blocks = re.findall(r'(diff --git[\s\S]+?)(?=\ndiff --git|\n```|\Z)', response)
    if not blocks:
        logger.warning("No diff --git block found in response")
        return response.strip()
    if len(blocks) > 1:
        logger.warning(f"Multiple diff blocks found ({len(blocks)}), selecting longest")
    # 选最长 (final patch 通常完整, partial 是 thinking 中的)
    longest = max(blocks, key=len)
    return longest.strip()

patch_match = extract_patch(response)

if patch_match:
    generated_patch = patch_match
    print(f"\n✓ Extracted patch ({len(generated_patch)} chars)")
else:
    generated_patch = response  # fallback
    print(f"\n⚠ No diff --git found, using full response")

# Save
with open(f"/tmp/swe-bench-day5/{TASK_ID}/generated_patch.diff", 'w') as f:
    f.write(generated_patch)

r1.unload()
