import tiktoken

def estimate_token_coverage(texts, encoding_name="o200k_base"):
    encoding = tiktoken.get_encoding(encoding_name)
    total_chars = 0
    total_diff_chars = 0

    for text in texts:
        tokens = encoding.encode(text)
        decoded = encoding.decode(tokens)
        total_chars += len(text)
        diff = sum(a != b for a, b in zip(text, decoded)) + abs(len(text) - len(decoded))
        total_diff_chars += diff

    return 1 - (total_diff_chars / total_chars) if total_chars > 0 else 0

def main():
    filename = "eval_text.txt"  # specify your file here

    with open(filename, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    coverage = estimate_token_coverage(texts)
    print(f"Estimated token coverage: {coverage:.4f}")

if __name__ == "__main__":
    main()
