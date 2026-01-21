"""
ROUGE Evaluation Script for Fine-tuned Nepali Summarization Model
Aligned with FineTunev2.1:
- Identical generation logic
- Reference-length-conditioned summaries
- Top-k + Top-p sampling
- Devanagari-only preprocessing before ROUGE
- Aggregate ROUGE reporting
"""

import os
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import sentencepiece as spm
from tqdm import tqdm
import csv
import re
from rouge_score import rouge_scorer
from rouge_score.tokenizers import Tokenizer

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class EvalConfig:
    MODEL_PATH: str = r"C:\Users\prash\Documents\AI\Major Project\FineTuning\finetuned-summarization\best_model.pt"
    TOKENIZER_PATH: str = r"C:\Users\prash\Documents\AI\Major Project\FineTuning\new-tokenizers\bpe-16-updated.model"
    TEST_JSONL: str = r"C:\Users\prash\Documents\AI\Major Project\FineTuning\nepali_XLSum_v2.0\Filtered\test_filtered.jsonl"
    OUTPUT_CSV: str = r"C:\Users\prash\Documents\AI\Major Project\FineTuning\finetuned-summarization\results\rouge_results.csv"

    BLOCK_SIZE: int = 1024
    TOP_K: int = 40
    TOP_P: float = 0.9
    TEMPERATURE: float = 0.6

PROMPT_TEMPLATE = "यो लेखको संक्षेप गर्नुहोस्:\n{text}\nसारांश:\n"

# ============================================================================
# MODEL ARCHITECTURE (IDENTICAL TO TRAINING)
# ============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 16384
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
                wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
                h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
                ln_f=nn.LayerNorm(cfg.n_embd),
            )
        )
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, input_ids):
        B, T = input_ids.size()
        pos = torch.arange(0, T, device=input_ids.device)
        x = self.transformer.wte(input_ids) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)

# ============================================================================
# PREPROCESSING
# ============================================================================

DEVANAGARI_REGEX = re.compile(r"[^\u0900-\u097F\s।]")

def preprocess_nepali(text: str) -> str:
    text = DEVANAGARI_REGEX.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ============================================================================
# ROUGE TOKENIZER
# ============================================================================

class NepaliTokenizer(Tokenizer):
    def tokenize(self, text):
        text = text.replace("।", "")
        return text.split()

def get_rouge_scorer():
    return rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        tokenizer=NepaliTokenizer(),
        use_stemmer=False,
    )

# ============================================================================
# GENERATION
# ============================================================================

def generate_summary(model, tokenizer, text, ref_summary, cfg, device):
    prompt = PROMPT_TEMPLATE.format(text=text)
    prompt_tokens = tokenizer.encode(prompt)
    ref_len = len(tokenizer.encode(ref_summary))

    input_ids = torch.tensor(prompt_tokens).unsqueeze(0).to(device)
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(ref_len):
            if generated.size(1) >= cfg.BLOCK_SIZE:
                break

            logits = model(generated)[:, -1, :] / cfg.TEMPERATURE
            top_k = min(cfg.TOP_K, logits.size(-1))
            logits, indices = torch.topk(logits, top_k)

            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)

            cutoff = cumulative > cfg.TOP_P
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False

            sorted_probs[cutoff] = 0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

            sampled = torch.multinomial(sorted_probs, 1)
            next_token = sorted_idx.gather(-1, sampled)
            next_token = indices.gather(-1, next_token)

            generated = torch.cat([generated, next_token], dim=1)

    return tokenizer.decode(generated[0, len(prompt_tokens):].tolist())

# ============================================================================
# MAIN EVALUATION
# ============================================================================

def evaluate():
    cfg = EvalConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sp = spm.SentencePieceProcessor()
    sp.load(cfg.TOKENIZER_PATH)

    checkpoint = torch.load(cfg.MODEL_PATH, map_location=device, weights_only=False)
    model = GPT(checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    test_data = [json.loads(l) for l in open(cfg.TEST_JSONL, encoding="utf-8")]

    scorer = get_rouge_scorer()

    os.makedirs(os.path.dirname(cfg.OUTPUT_CSV), exist_ok=True)
    csv_file = open(cfg.OUTPUT_CSV, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)

    writer.writerow([
        "id",
        "r1_p", "r1_r", "r1_f",
        "r2_p", "r2_r", "r2_f",
        "rL_p", "rL_r", "rL_f",
        "reference", "generated"
    ])

    agg = {m: {"p": [], "r": [], "f": []} for m in ["rouge1", "rouge2", "rougeL"]}

    for idx, item in enumerate(tqdm(test_data, desc="Evaluating")):
        gen = generate_summary(model, sp, item["text"], item["summary"], cfg, device)
        ref = preprocess_nepali(item["summary"])
        gen = preprocess_nepali(gen)

        scores = scorer.score(ref, gen)

        for m in agg:
            agg[m]["p"].append(scores[m].precision)
            agg[m]["r"].append(scores[m].recall)
            agg[m]["f"].append(scores[m].fmeasure)

        writer.writerow([
            idx,
            scores["rouge1"].precision, scores["rouge1"].recall, scores["rouge1"].fmeasure,
            scores["rouge2"].precision, scores["rouge2"].recall, scores["rouge2"].fmeasure,
            scores["rougeL"].precision, scores["rougeL"].recall, scores["rougeL"].fmeasure,
            ref, gen
        ])

    def avg(x): return sum(x) / len(x)

    writer.writerow([])
    writer.writerow(["# AGGREGATE ROUGE SCORES"])
    for m in agg:
        writer.writerow([f"# {m.upper()}", avg(agg[m]["p"]), avg(agg[m]["r"]), avg(agg[m]["f"])])

    csv_file.close()

    print("\nFINAL AGGREGATE ROUGE SCORES")
    print("=" * 50)
    for m in agg:
        print(f"{m.upper():7s} | F1: {avg(agg[m]['f']) * 100:.2f}")
    print("=" * 50)

if __name__ == "__main__":
    evaluate()
