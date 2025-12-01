#!/usr/bin/env python3
import sentencepiece as smp
import tiktoken
import os
import csv
import pandas as pd

# CONFIGURATION
text_file = "eval_text.txt"
output_file = 'tokenizer_evaluation.csv'
dirs_to_search = ['bpe-token-models', 'word-token-models', 'uni-token-models']

def evaluate_model(model_path, eval_texts):
    try:
        sp = smp.SentencePieceProcessor(model_file=model_path)
       
        total_words = total_tokens = unk_tokens = 0
        all_chars = covered_chars = set()
       
        for text in eval_texts:
            words = text.split()
            token_ids = sp.encode(text)
           
            total_words += len(words)
            total_tokens += len(token_ids)
           
            # Character coverage
            all_chars.update(set(text))
            decoded = sp.decode(token_ids)
            covered_chars.update(char for char in text if char in decoded)
           
            # UNK tokens
            unk_id = sp.unk_id()
            if unk_id != -1:
                unk_tokens += token_ids.count(unk_id)
       
        return {
            'vocab_size': sp.get_piece_size(),
            'token_word_ratio': total_tokens / total_words if total_words > 0 else 0,
            'character_coverage': len(covered_chars) / len(all_chars) if all_chars else 0,
            'token_coverage': 1 - (unk_tokens / total_tokens) if total_tokens > 0 else 0,
            'status': 'success'
        }
    except Exception as e:
        return {'vocab_size': 0, 'token_word_ratio': 0, 'character_coverage': 0,
                'token_coverage': 0, 'status': f'error: {str(e)}'}

def evaluate_tiktoken_model(eval_texts):
    """Evaluate tiktoken o200k_base model using same logic as SentencePiece"""
    try:
        tokenizer = tiktoken.get_encoding("o200k_base")
        
        total_words = total_tokens = 0
        all_chars = covered_chars = set()
        
        for text in eval_texts:
            words = text.split()
            token_ids = tokenizer.encode(text)
            
            total_words += len(words)
            total_tokens += len(token_ids)
            
            # Character coverage
            all_chars.update(set(text))
            decoded = tokenizer.decode(token_ids)
            covered_chars.update(char for char in text if char in decoded)
        
        # tiktoken doesn't have explicit UNK tokens like SentencePiece
        # Assume high token coverage for o200k_base
        token_coverage = 1.0
        
        return {
            'vocab_size': tokenizer.n_vocab,
            'token_word_ratio': total_tokens / total_words if total_words > 0 else 0,
            'character_coverage': len(covered_chars) / len(all_chars) if all_chars else 0,
            'token_coverage': token_coverage,
            'status': 'success'
        }
    except Exception as e:
        return {'vocab_size': 0, 'token_word_ratio': 0, 'character_coverage': 0,
                'token_coverage': 0, 'status': f'error: {str(e)}'}

def main():
    # Load texts
    if os.path.exists(text_file):
        with open(text_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        texts = ["नमस्कार! यो एक परीक्षण वाक्य हो।", "मशीन लर्निङ र प्राकृतिक भाषा प्रशोधन।"]
   
    # Find SentencePiece models
    results = []
    for base_dir in dirs_to_search:
        if os.path.exists(base_dir):
            for file in os.listdir(base_dir):
                if file.endswith('.model'):
                    model_path = os.path.join(base_dir, file)
                    metrics = evaluate_model(model_path, texts)
                   
                    results.append({
                        'Model_Folder': base_dir,
                        'Model_Name': os.path.splitext(file)[0],
                        'Vocab_Size': metrics['vocab_size'],
                        'Token_Word_Ratio': round(metrics['token_word_ratio'], 4),
                        'Character_Coverage': round(metrics['character_coverage'], 4),
                        'Token_Coverage': round(metrics['token_coverage'], 4),
                        'Status': metrics['status']
                    })
    
    # Add tiktoken o200k_base evaluation
    tiktoken_metrics = evaluate_tiktoken_model(texts)
    results.append({
        'Model_Folder': 'tiktoken',
        'Model_Name': 'o200k_base',
        'Vocab_Size': tiktoken_metrics['vocab_size'],
        'Token_Word_Ratio': round(tiktoken_metrics['token_word_ratio'], 4),
        'Character_Coverage': round(tiktoken_metrics['character_coverage'], 4),
        'Token_Coverage': round(tiktoken_metrics['token_coverage'], 4),
        'Status': tiktoken_metrics['status']
    })
   
    # Save to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Model_Folder', 'Model_Name', 'Vocab_Size',
                                                    'Token_Word_Ratio', 'Character_Coverage',
                                                    'Token_Coverage', 'Status'])
        writer.writeheader()
        writer.writerows(results)
   
    print(f"Results saved to {output_file}\n")
   
    view = pd.read_csv("tokenizer_evaluation.csv")
    print(view.head(10))

if __name__ == "__main__":
    main()