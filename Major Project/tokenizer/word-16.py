import sentencepiece as spm

def train_sentencepiece_tokenizer(
    input_file=r"/home/basanta/BPE/data_preparation/cleaned_text.txt",
    model_prefix="word-16",
    vocab_size=16384,
    character_coverage=0.9995,
    input_sentence_size=8000000,
    shuffle_input_sentence=True,
    train_extremely_large_corpus=True,
    num_threads=4,
    max_sentence_length=8192,
    model_type="word"
):
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=shuffle_input_sentence,
        train_extremely_large_corpus=train_extremely_large_corpus,
        num_threads=num_threads,
        max_sentence_length=max_sentence_length,
        bos_id=1,
        eos_id=2,
        unk_id=0,
        pad_id=3
    )
    print(f"✅ Tokenizer trained and saved as: {model_prefix}.model and {model_prefix}.vocab")

if __name__ == "__main__":
    train_sentencepiece_tokenizer()