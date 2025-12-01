import re

def clean_non_devanagari(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove CSS-like blocks
    text = re.sub(r'\{[^}]*\}', '', text)
    # Remove double danda (॥)
    text = re.sub(r'\u0965', '', text)
    # Remove non-Devanagari characters except digits and danda
    allowed_pattern = r'[^\u0900-\u097F\u0966-\u096F\u0964\s]'
    text = re.sub(allowed_pattern, '', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

input_path = "combined_output.txt"
output_path = "cleaned_text.txt"

with open(input_path, "r", encoding="utf-8") as infile, \
     open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        cleaned = clean_non_devanagari(line)
        if cleaned:  # skip empty lines
            outfile.write(cleaned + "\n")
