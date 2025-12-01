import os
import multiprocessing as mp
import numpy as np
import tiktoken
from tqdm import tqdm # pip install tqdm

# ------------------------------------------
local_dir = "shards-base"
input_file = r"/home/basanta/BPE/data_preparation/cleaned_text.txt"  # Path to your local data file
shard_size = int(1e7)  # 10M tokens per shard

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# init the tokenizer
enc = tiktoken.get_encoding("o200k_base")
eot = enc._special_tokens['<|endoftext|>']  # end of text token

def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint32 tokens
    tokens = [eot]  # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**32).all(), "token dictionary too large for uint32"
    tokens_np_uint32 = tokens_np.astype(np.uint32)
    return tokens_np_uint32

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

def read_local_data(file_path):
    # Reads local data file line by line
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield line.strip()

# Tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
if __name__ == '__main__':
    # Tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint32)
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(tokenize, read_local_data(input_file), chunksize=16):
            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"nepberta_{split}_{shard_index:06d}")
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        # write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"nepberta_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])