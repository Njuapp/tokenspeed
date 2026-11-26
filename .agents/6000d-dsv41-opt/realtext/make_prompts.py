"""Build N distinct real-text prompts of exactly 8192 tokens for DeepSeek-V4.1-Flash.

Text comes from the public ShareGPT V3 dump (consecutive conversation turns of shuffled
conversations, seed 42, concatenated until long enough). Each prompt is one chat user turn in
the official DeepSeek-V4.1 encoding (non-thinking chat mode):
  <｜begin▁of▁sentence｜><｜User｜> CONTENT <｜Assistant｜></think>
with CONTENT cut at token level so the total is exactly 8192 tokens.
Runs inside the SGLang image (needs transformers). Output schema: {"prompts": [{ordinal, prompt_token_ids, input_ids_sha256}]}.
"""

import argparse
import hashlib
import json
import random

from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--sharegpt", required=True)
p.add_argument("--out", required=True)
p.add_argument("--n", type=int, default=160)
p.add_argument("--length", type=int, default=8192)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

try:
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
except (
    Exception
):  # config.json architecture unknown to this transformers: load tokenizer.json directly
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast(tokenizer_file=f"{a.model}/tokenizer.json")
ids = lambda s: tok.convert_tokens_to_ids(s)
prefix = [ids("<｜begin▁of▁sentence｜>"), ids("<｜User｜>")]
suffix = [ids("<｜Assistant｜>"), ids("</think>")]
assert None not in prefix + suffix and tok.unk_token_id not in prefix + suffix, (
    prefix,
    suffix,
)
need = a.length - len(prefix) - len(suffix)

convs = json.load(open(a.sharegpt))
random.Random(a.seed).shuffle(convs)
texts = (
    t.get("value", "")
    for c in convs
    for t in c.get("conversations", [])
    if t.get("value")
)
prompts = []
while len(prompts) < a.n:
    content = []
    while len(content) < need:
        content += tok(next(texts) + "\n\n", add_special_tokens=False)["input_ids"]
    full = prefix + content[:need] + suffix
    assert len(full) == a.length
    prompts.append(
        {
            "ordinal": len(prompts),
            "prompt_token_ids": full,
            "input_ids_sha256": hashlib.sha256(json.dumps(full).encode()).hexdigest(),
        }
    )
json.dump(
    {
        "schema_version": "dsv41-sharegpt-8k-v1",
        "source": "ShareGPT_V3_unfiltered_cleaned_split.json",
        "seed": a.seed,
        "template_prefix_ids": prefix,
        "template_suffix_ids": suffix,
        "prompts": prompts,
    },
    open(a.out, "w"),
)
print(
    f"wrote {len(prompts)} prompts x {a.length} tokens to {a.out}; prefix {prefix} suffix {suffix}"
)
