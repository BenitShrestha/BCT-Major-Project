import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load('nepali_bpe_sample.model')

sentence = "विद्यार्थीहरूले परीक्षाको तयारी गरिरहेका छन् । अत्यधिक मेहेनत गर्दा सफलता प्राप्त गर्न सकिन्छ ।"
encoded = sp.encode(sentence, out_type=str)
print("Encoded:", len(encoded),encoded)

