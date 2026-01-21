"""
ROUGE Evaluation Script for Fine-tuned Nepali Summarization Model
Generates summaries and computes ROUGE scores on entire test set
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
    # Paths
    MODEL_PATH: str = r"C:\Users\prash\Documents\AI\Major Project\FineTuning\finetuned-summarization\best_model.pt"  # or epoch_X.pt
    TOKENIZER_PATH: str = r"C:\Users\prash\Documents\AI\Major Project\FineTuning\new-tokenizers\bpe-16-updated.model"
    TEST_JSONL: str = r"C:\Users\prash\Documents\AI\Major Project\FineTuning\nepali_XLSum_v2.0\Filtered\test_filtered.jsonl"
    OUTPUT_CSV: str = r"C:\Users\prash\Documents\AI\Major Project\FineTuning\finetuned-summarization\results\rouge_results.csv"
    
    # Generation settings
    BLOCK_SIZE: int = 1024
    MAX_NEW_TOKENS: int = 64
    TOP_K: int = 40
    TEMPERATURE: float = 0.7

PROMPT_TEMPLATE = "यो लेखको संक्षेप गर्नुहोस्:\n{text}\nसारांश:\n"

# ============================================================================
# MODEL ARCHITECTURE (Same as training script)
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
        self.gelu = nn.GELU(approximate='tanh')
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
        self.config = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, input_ids):
        B, T = input_ids.size()
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        x = self.transformer.wte(input_ids) + self.transformer.wpe(pos)
        
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        return self.lm_head(x)

# ============================================================================
# NEPALI ROUGE SCORER
# ============================================================================

class NepaliTokenizer(Tokenizer):
    def tokenize(self, text):
        text = text.replace("।", "")
        text = re.sub(r"[^\w\s]", "", text)
        return text.split()

def get_rouge_scorer():
    return rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        tokenizer=NepaliTokenizer(),
        use_stemmer=False
    )

# ============================================================================
# GENERATION
# ============================================================================

def generate_summary(model, tokenizer, text, cfg, device):
    """Generate summary using top-k sampling"""
    model.eval()
    prompt = PROMPT_TEMPLATE.format(text=text)
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(cfg.MAX_NEW_TOKENS):
            if generated.size(1) >= cfg.BLOCK_SIZE:
                break
            
            logits = model(generated)
            logits = logits[:, -1, :] / cfg.TEMPERATURE
            
            top_k_logits, top_k_indices = torch.topk(logits, min(cfg.TOP_K, logits.size(-1)))
            probs = F.softmax(top_k_logits, dim=-1)
            next_token = torch.gather(top_k_indices, -1, torch.multinomial(probs, 1))
            
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == 0:  # EOS token
                break
    
    return tokenizer.decode(generated[0, len(tokens):].tolist())

# ============================================================================
# MAIN EVALUATION
# ============================================================================

def evaluate():
    cfg = EvalConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*80)
    print("ROUGE EVALUATION ON TEST SET")
    print("="*80)
    print(f"Device: {device}\n")
    
    # Load tokenizer
    print("Loading tokenizer...")
    sp = spm.SentencePieceProcessor()
    sp.load(cfg.TOKENIZER_PATH)
    print(f"✓ Tokenizer loaded (vocab: {sp.vocab_size()})\n")
    
    # Load model
    print(f"Loading model from: {cfg.MODEL_PATH}")
    checkpoint = torch.load(cfg.MODEL_PATH, map_location=device, weights_only=False)
    model = GPT(checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    print(f"✓ Model loaded")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}, Step: {checkpoint['global_step']}")
    if 'eval_loss' in checkpoint:
        print(f"  Eval loss: {checkpoint['eval_loss']:.4f}\n")
    
    # Load test data
    print(f"Loading test data from: {cfg.TEST_JSONL}")
    test_data = []
    with open(cfg.TEST_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if 'text' in item and 'summary' in item:
                    test_data.append(item)
    print(f"✓ Loaded {len(test_data)} test samples\n")
    
    # Initialize ROUGE scorer
    scorer = get_rouge_scorer()
    
    # Prepare CSV
    csv_file = open(cfg.OUTPUT_CSV, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'sample_id', 
        'rouge1_precision', 'rouge1_recall', 'rouge1_fmeasure',
        'rouge2_precision', 'rouge2_recall', 'rouge2_fmeasure',
        'rougeL_precision', 'rougeL_recall', 'rougeL_fmeasure',
        'reference_summary', 'generated_summary'
    ])
    
    # Accumulate scores
    all_scores = {
        'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
        'rouge2': {'precision': [], 'recall': [], 'fmeasure': []},
        'rougeL': {'precision': [], 'recall': [], 'fmeasure': []}
    }
    
    print("="*80)
    print("GENERATING SUMMARIES AND COMPUTING ROUGE")
    print("="*80)
    
    # Process each test sample
    for idx, item in enumerate(tqdm(test_data, desc="Evaluating")):
        # Generate summary
        generated = generate_summary(model, sp, item['text'], cfg, device)
        reference = item['summary']
        
        # Compute ROUGE scores
        scores = scorer.score(reference, generated)
        
        # Write to CSV
        csv_writer.writerow([
            idx,
            scores['rouge1'].precision, scores['rouge1'].recall, scores['rouge1'].fmeasure,
            scores['rouge2'].precision, scores['rouge2'].recall, scores['rouge2'].fmeasure,
            scores['rougeL'].precision, scores['rougeL'].recall, scores['rougeL'].fmeasure,
            reference, generated
        ])
        
        # Accumulate
        for metric in ['rouge1', 'rouge2', 'rougeL']:
            all_scores[metric]['precision'].append(getattr(scores[metric], 'precision'))
            all_scores[metric]['recall'].append(getattr(scores[metric], 'recall'))
            all_scores[metric]['fmeasure'].append(getattr(scores[metric], 'fmeasure'))
    
    csv_file.close()
    
    # Calculate averages
    print("\n" + "="*80)
    print("FINAL RESULTS (Averaged over entire test set)")
    print("="*80)
    
    for metric in ['rouge1', 'rouge2', 'rougeL']:
        avg_p = sum(all_scores[metric]['precision']) / len(all_scores[metric]['precision']) * 100
        avg_r = sum(all_scores[metric]['recall']) / len(all_scores[metric]['recall']) * 100
        avg_f = sum(all_scores[metric]['fmeasure']) / len(all_scores[metric]['fmeasure']) * 100
        
        print(f"\n{metric.upper()}:")
        print(f"  Precision: {avg_p:.2f}")
        print(f"  Recall:    {avg_r:.2f}")
        print(f"  F-measure: {avg_f:.2f}")
    
    print(f"\n{'='*80}")
    print(f"✓ Detailed results saved to: {cfg.OUTPUT_CSV}")
    print(f"✓ Total samples evaluated: {len(test_data)}")
    print("="*80)

if __name__ == "__main__":
    evaluate()