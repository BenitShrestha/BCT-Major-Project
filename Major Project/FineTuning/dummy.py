# ============================================================================
# GPT-2 Fine-tuning for Nepali Summarization - Improved Version
# ============================================================================

import os
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from tqdm import tqdm
import sentencepiece as spm
from datetime import datetime
import math

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

class Config:
    # Paths
    CHECKPOINT_PATH = r"C:\Users\prash\Documents\AI\Major Project\FineTuning\new-models\bpe16_updated_19k.pt"
    TOKENIZER_PATH = r"C:\Users\prash\Documents\AI\Major Project\FineTuning\new-tokenizers\bpe-16-updated.model"
    TRAIN_JSONL = r"C:\Users\prash\Documents\AI\Major Project\FineTuning\nepali_XLSum_v2.0\train_plus_val_norm.jsonl"  # Updated to normalized dataset
    TEST_JSONL = r"C:\Users\prash\Documents\AI\Major Project\FineTuning\nepali_XLSum_v2.0\test_norm.jsonl"  # Updated to normalized dataset
    OUTPUT_DIR = "finetuned-summarization"
    
    # Model settings
    BLOCK_SIZE = 1024  # Model hard limit
    MAX_SOURCE_LENGTH = 768  # 75% of block size for article + prompt
    MAX_TARGET_LENGTH = 256  # 25% of block size for summary
    
    # Freezing strategy: CORRECTED - No embedding freeze to avoid weight tying issues
    FREEZE_EMBEDDINGS = False  # Keeps both wte and lm_head trainable
    FREEZE_BLOCKS = list(range(8))  # Freeze first 8 blocks (0-7), train last 4 (8-11)
    
    # Training hyperparameters - IMPROVED
    NUM_EPOCHS = 5
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 2
    LEARNING_RATE = 1e-5  # REDUCED from 3e-5 for stability
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0
    WEIGHT_DECAY = 0.01
    
    # Generation settings - NEW
    GENERATION_TOP_K = 40
    GENERATION_TEMPERATURE = 0.7
    
    # Evaluation
    EVAL_STEPS = 100
    
    # Safety settings - NEW
    MAX_CONSECUTIVE_NANS = 3  # Stop if 3 consecutive NaN batches

config = Config()
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# 2. MODEL ARCHITECTURE
# ============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

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
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, input_ids, labels=None):
        B, T = input_ids.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"
        
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(input_ids)
        x = tok_emb + pos_emb
        
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return loss, logits
    
# ============================================================================
# 3. LOAD MODEL & APPLY FREEZING
# ============================================================================

print("\n" + "="*80)
print("LOADING MODEL")
print("="*80)

sp = spm.SentencePieceProcessor()
sp.load(config.TOKENIZER_PATH)
print(f"✓ Tokenizer loaded (vocab: {sp.vocab_size()})")

checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=device, weights_only=False)
model_config = checkpoint['config']
model = GPT(model_config)
model.load_state_dict(checkpoint['model'])
model.to(device)
print(f"✓ Model loaded (step: {checkpoint['step']}, val_loss: {checkpoint['val_loss']:.4f})")

# Apply freezing strategy
print("\n" + "="*80)
print("APPLYING FREEZING STRATEGY")
print("="*80)

# Only freeze position embeddings (token embeddings remain trainable with lm_head)
if config.FREEZE_EMBEDDINGS:
    for param in model.transformer.wte.parameters():
        param.requires_grad = False
    for param in model.transformer.wpe.parameters():
        param.requires_grad = False
    print("✓ Frozen: Token & Position Embeddings")
else:
    # Only freeze position embeddings
    for param in model.transformer.wpe.parameters():
        param.requires_grad = False
    print("✓ Frozen: Position Embeddings only")
    print("✓ Token Embeddings (wte) remain trainable (shares weights with lm_head)")

# Freeze specified transformer blocks
for block_idx in config.FREEZE_BLOCKS:
    for param in model.transformer.h[block_idx].parameters():
        param.requires_grad = False
print(f"✓ Frozen: Blocks {config.FREEZE_BLOCKS[0]}-{config.FREEZE_BLOCKS[-1]}")
print(f"✓ Trainable: Blocks {config.FREEZE_BLOCKS[-1]+1}-{model_config.n_layer-1} + Final LayerNorm + LM Head")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"\n✓ Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

# ============================================================================
# 4. DATASET PREPARATION
# ============================================================================

print("\n" + "="*80)
print("LOADING DATASET")
print("="*80)

# UPDATED: Nepali prompt template
PROMPT_TEMPLATE = "यो लेखको संक्षेप गर्नुहोस्:\n{text}\nसारांश:\n"

class SummarizationDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if 'text' in item and 'summary' in item:
                        self.data.append(item)
        
        print(f"  Loaded {len(self.data)} samples from {os.path.basename(jsonl_path)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Build full sequence (normalized dataset should already fit in block_size)
        prompt = PROMPT_TEMPLATE.format(text=item['text'])
        full_text = prompt + item['summary']
        
        # Tokenize
        tokens = self.tokenizer.encode(full_text)
        
        # Safety check: truncate if still exceeds (shouldn't happen with normalized data)
        if len(tokens) > self.block_size:
            print(f"  ⚠️  Sample {idx} exceeds block_size ({len(tokens)} > {self.block_size}), truncating")
            tokens = tokens[:self.block_size]
        
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        
        # Mask prompt tokens in labels (only compute loss on summary)
        prompt_len = len(self.tokenizer.encode(prompt))
        labels[:prompt_len] = -100
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'text': item['text'],
            'summary': item['summary']
        }

train_dataset = SummarizationDataset(config.TRAIN_JSONL, sp, config.BLOCK_SIZE)
test_dataset = SummarizationDataset(config.TEST_JSONL, sp, config.BLOCK_SIZE)

print(f"\n✓ Train samples: {len(train_dataset)}")
print(f"✓ Test samples:  {len(test_dataset)}")

def collate_fn(batch):
    max_len = max(len(item['input_ids']) for item in batch)
    
    input_ids = []
    labels = []
    
    for item in batch:
        seq_len = len(item['input_ids'])
        pad_len = max_len - seq_len
        
        input_ids.append(torch.cat([item['input_ids'], torch.zeros(pad_len, dtype=torch.long)]))
        labels.append(torch.cat([item['labels'], torch.full((pad_len,), -100, dtype=torch.long)]))
    
    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels)
    }

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print(f"✓ Train batches per epoch: {len(train_loader)}")
print(f"✓ Optimizer steps per epoch: ~{len(train_loader) // config.GRADIENT_ACCUMULATION_STEPS}")

# ============================================================================
# 5. OPTIMIZER & SCHEDULER
# ============================================================================

print("\n" + "="*80)
print("OPTIMIZER SETUP")
print("="*80)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=config.LEARNING_RATE,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=config.WEIGHT_DECAY
)

total_steps = config.NUM_EPOCHS * len(train_loader) // config.GRADIENT_ACCUMULATION_STEPS
warmup_steps = int(config.WARMUP_RATIO * total_steps)

def get_lr(step):
    if step < warmup_steps:
        return config.LEARNING_RATE * (step + 1) / warmup_steps
    if step >= total_steps:
        return config.LEARNING_RATE * 0.1
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.LEARNING_RATE * 0.1 + coeff * (config.LEARNING_RATE * 0.9)

print(f"✓ Optimizer: AdamW")
print(f"✓ Learning rate: {config.LEARNING_RATE} (REDUCED for stability)")
print(f"✓ Weight decay: {config.WEIGHT_DECAY}")
print(f"✓ Total training steps: {total_steps}")
print(f"✓ Warmup steps: {warmup_steps}")
print(f"✓ Gradient accumulation: {config.GRADIENT_ACCUMULATION_STEPS}")
print(f"✓ Effective batch size: {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS}")

# ============================================================================
# 6. EVALUATION FUNCTIONS
# ============================================================================

def evaluate_loss(model, dataloader):
    """Compute average loss on evaluation set"""
    model.eval()
    total_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            try:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                loss, _ = model(input_ids, labels)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item()
                    count += 1
            except RuntimeError as e:
                print(f"  ⚠️  Error during evaluation: {e}")
                continue
    
    model.train()
    return total_loss / count if count > 0 else float('nan')

def generate_summary_topk(model, tokenizer, text, max_new_tokens=128, top_k=40, temperature=0.7):
    """Generate summary using top-k sampling (IMPROVED)"""
    model.eval()
    prompt = PROMPT_TEMPLATE.format(text=text)
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if generated.size(1) >= model.config.block_size:
                break
            
            _, logits = model(generated)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)
            next_token_idx = torch.multinomial(probs, 1)
            next_token = torch.gather(top_k_indices, -1, next_token_idx)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop at EOS or padding token
            if next_token.item() == 0:
                break
    
    summary = tokenizer.decode(generated[0, len(tokens):].tolist())
    model.train()
    return summary

def generate_summary_greedy(model, tokenizer, text, max_new_tokens=128):
    """Generate summary using greedy decoding (fallback)"""
    model.eval()
    prompt = PROMPT_TEMPLATE.format(text=text)
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if generated.size(1) >= model.config.block_size:
                break
            
            _, logits = model(generated)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            if next_token.item() == 0:
                break
    
    summary = tokenizer.decode(generated[0, len(tokens):].tolist())
    model.train()
    return summary

def compute_rouge(model, dataset, sample_size=None, use_topk=True):
    """Compute ROUGE scores on dataset"""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("⚠️  rouge_score not installed. Run: pip install rouge-score")
        return None
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    # Use sample if dataset is large
    eval_size = sample_size if sample_size and sample_size < len(dataset) else len(dataset)
    
    print(f"  Computing ROUGE on {eval_size} samples (method: {'top-k' if use_topk else 'greedy'})...")
    for i in tqdm(range(eval_size), desc="ROUGE"):
        item = dataset.data[i]
        
        if use_topk:
            pred = generate_summary_topk(
                model, sp, item['text'], 
                config.MAX_TARGET_LENGTH,
                config.GENERATION_TOP_K,
                config.GENERATION_TEMPERATURE
            )
        else:
            pred = generate_summary_greedy(model, sp, item['text'], config.MAX_TARGET_LENGTH)
        
        ref = item['summary']
        
        result = scorer.score(ref, pred)
        scores['rouge1'].append(result['rouge1'].fmeasure)
        scores['rouge2'].append(result['rouge2'].fmeasure)
        scores['rougeL'].append(result['rougeL'].fmeasure)
    
    return {k: sum(v) / len(v) * 100 for k, v in scores.items()}

# ============================================================================
# 7. TRAINING LOOP
# ============================================================================

print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)

global_step = 0
best_eval_loss = float('inf')
history = []
consecutive_nans = 0  # Track consecutive NaN batches

for epoch in range(config.NUM_EPOCHS):
    print(f"\n{'='*80}")
    print(f"EPOCH {epoch + 1}/{config.NUM_EPOCHS}")
    print(f"{'='*80}")
    
    model.train()
    epoch_loss = 0.0
    optimizer.zero_grad()
    batches_processed = 0
    
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        loss, _ = model(input_ids, labels)
        
        # Check for NaN or Inf
        if torch.isnan(loss) or torch.isinf(loss):
            consecutive_nans += 1
            print(f"  ⚠️  {'NaN' if torch.isnan(loss) else 'Inf'} loss detected at step {global_step} (consecutive: {consecutive_nans})")
            
            if consecutive_nans >= config.MAX_CONSECUTIVE_NANS:
                print(f"  ❌ Too many consecutive NaN/Inf losses ({consecutive_nans}). Stopping training.")
                print(f"  💡 Suggestions:")
                print(f"     - Check if normalized dataset has issues")
                print(f"     - Further reduce learning rate")
                print(f"     - Check for data corruption")
                break
            
            optimizer.zero_grad()
            continue
        
        # Reset NaN counter on successful batch
        consecutive_nans = 0
        
        # Backward pass with gradient accumulation
        loss = loss / config.GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        
        epoch_loss += loss.item()
        batches_processed += 1
        
        # Optimizer step after accumulation
        if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            
            # Check for exploding gradients
            if grad_norm > config.MAX_GRAD_NORM * 10:
                print(f"  ⚠️  Large gradient norm detected: {grad_norm:.2f}")
            
            lr = get_lr(global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            
            # Logging
            if global_step % 10 == 0:
                current_loss = loss.item() * config.GRADIENT_ACCUMULATION_STEPS
                print(f"  Step {global_step:4d} | Loss: {current_loss:.4f} | LR: {lr:.2e} | Grad: {grad_norm:.2f}")
            
            # Step-wise evaluation
            if global_step % config.EVAL_STEPS == 0:
                print(f"\n  {'─'*76}")
                print(f"  EVALUATION AT STEP {global_step}")
                print(f"  {'─'*76}")
                
                eval_loss = evaluate_loss(model, test_loader)
                print(f"  → Eval loss: {eval_loss:.4f}")
                
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    print(f"  → ✨ New best eval loss!")
                    
                    # Save best model
                    best_model_path = os.path.join(config.OUTPUT_DIR, "best_model.pt")
                    torch.save({
                        'model': model.state_dict(),
                        'config': model.config,
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch + 1,
                        'global_step': global_step,
                        'eval_loss': eval_loss,
                    }, best_model_path)
                    print(f"  → Saved best model: {best_model_path}")
                
                print(f"  {'─'*76}\n")
        
        # Clear CUDA cache periodically
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()
    
    # Check if training was stopped due to NaNs
    if consecutive_nans >= config.MAX_CONSECUTIVE_NANS:
        print(f"\n❌ Training stopped early at epoch {epoch + 1} due to instability")
        break
    
    # End of epoch evaluation
    print(f"\n{'-'*80}")
    print(f"EPOCH {epoch + 1} SUMMARY")
    print(f"{'-'*80}")
    
    avg_train_loss = (epoch_loss * config.GRADIENT_ACCUMULATION_STEPS) / batches_processed if batches_processed > 0 else float('nan')
    eval_loss = evaluate_loss(model, test_loader)
    
    print(f"Average train loss: {avg_train_loss:.4f}")
    print(f"Evaluation loss:    {eval_loss:.4f}")
    
    # Compute ROUGE scores (use top-k sampling)
    rouge_scores = compute_rouge(model, test_dataset, sample_size=100, use_topk=True)
    if rouge_scores:
        print(f"\nROUGE Scores (on 100 test samples with top-k sampling):")
        print(f"  ROUGE-1: {rouge_scores['rouge1']:.2f}")
        print(f"  ROUGE-2: {rouge_scores['rouge2']:.2f}")
        print(f"  ROUGE-L: {rouge_scores['rougeL']:.2f}")
    
    # Save epoch checkpoint
    checkpoint_path = os.path.join(config.OUTPUT_DIR, f"epoch_{epoch + 1}.pt")
    torch.save({
        'model': model.state_dict(),
        'config': model.config,
        'optimizer': optimizer.state_dict(),
        'epoch': epoch + 1,
        'global_step': global_step,
        'train_loss': avg_train_loss,
        'eval_loss': eval_loss,
        'rouge_scores': rouge_scores,
    }, checkpoint_path)
    print(f"\n✓ Checkpoint saved: {checkpoint_path}")
    
    # Update history
    history.append({
        'epoch': epoch + 1,
        'global_step': global_step,
        'train_loss': avg_train_loss,
        'eval_loss': eval_loss,
        'rouge_scores': rouge_scores,
    })
    
    print(f"{'='*80}\n")

# ============================================================================
# 8. FINAL EVALUATION & SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("FINAL EVALUATION ON FULL TEST SET")
print("="*80)

final_rouge = compute_rouge(model, test_dataset, use_topk=True)  # Full test set with top-k
if final_rouge:
    print(f"\nFinal ROUGE Scores (with top-k sampling):")
    print(f"  ROUGE-1: {final_rouge['rouge1']:.2f}")
    print(f"  ROUGE-2: {final_rouge['rouge2']:.2f}")
    print(f"  ROUGE-L: {final_rouge['rougeL']:.2f}")

# Save training log
log_path = os.path.join(config.OUTPUT_DIR, "training_log.json")
with open(log_path, 'w', encoding='utf-8') as f:
    json.dump({
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'freeze_embeddings': config.FREEZE_EMBEDDINGS,
            'freeze_blocks': config.FREEZE_BLOCKS,
            'trainable_params': trainable,
            'total_params': total,
            'trainable_percentage': 100 * trainable / total,
            'num_epochs': config.NUM_EPOCHS,
            'batch_size': config.BATCH_SIZE,
            'gradient_accumulation': config.GRADIENT_ACCUMULATION_STEPS,
            'learning_rate': config.LEARNING_RATE,
            'warmup_ratio': config.WARMUP_RATIO,
            'generation_top_k': config.GENERATION_TOP_K,
            'generation_temperature': config.GENERATION_TEMPERATURE,
        },
        'training_history': history,
        'final_rouge_scores': final_rouge,
        'best_eval_loss': best_eval_loss,
    }, f, indent=2, ensure_ascii=False)

print(f"\n✓ Training log saved: {log_path}")