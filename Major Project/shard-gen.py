import os
import multiprocessing as mp
import numpy as np
import sentencepiece as spm
from tqdm import tqdm

# ------------------------------------------
local_dir = "shards-word-50"
input_file = r"/home/basanta/BPE/data_preparation/cleaned_text.txt"  # Path to your local data file
shard_size = int(1e7)  # 10 million tokens per shard

# create the cache directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Load SentencePiece tokenizer
sp = spm.SentencePieceProcessor()
sp.load(r"/home/basanta/BPE/word-token-models/word-50.model")

# Get the <eos> token ID or define your own special token
eot = sp.piece_to_id("</s>") if sp.piece_to_id("</s>") != 0 else sp.piece_to_id("<unk>")

def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot]  # Add special end-of-text token at start
    tokens.extend(sp.encode(doc, out_type=int))
    tokens_np = np.array(tokens, dtype=np.uint32)
    tokens_np_uint16 = tokens_np.astype(np.uint16)  # Assuming vocab size < 65536
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

def read_local_data(file_path):
    # Reads local data file line by line
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield line.strip()

if __name__ == '__main__':
    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint32)
        token_count = 0
        progress_bar = None

        for tokens in pool.imap(tokenize, read_local_data(input_file), chunksize=16):
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                split = "val" if shard_index < 5 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"nepberta_{split}_{shard_index:06d}")
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        # write remaining tokens
        if token_count != 0:
            split = "val" if shard_index < 5 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"nepberta_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])
