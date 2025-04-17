
import os
import re

# Set file paths
original_de_path = "Data/original data/europarl-v7.de.csv"
original_en_path = "Data/original data/europarl-v7.en.csv"
output_de_path = "Data/filtered_de.csv"
output_en_path = "Data/filtered_en.csv"

# Cleaning function: keep only letters, numbers, spaces, and hyphens
def remove_punctuation_except_hyphen(text):
    return ''.join(
        ch for ch in text
        if ch.isalnum() or ch.isspace() or ch == '-'
    )

# Check if a sentence starts with a non-letter character (e.g., punctuation)
def starts_with_punctuation(text):
    return bool(re.match(r"^\W", text.strip(), re.UNICODE))

# Read original files
with open(original_de_path, 'r', encoding='utf-8') as f:
    full_de_lines = [line.strip() for line in f.readlines()]
with open(original_en_path, 'r', encoding='utf-8') as f:
    full_en_lines = [line.strip() for line in f.readlines()]

# Strict one-to-one alignment: both sides must contain one or fewer periods
strict_full_de = []
strict_full_en = []
min_len = min(len(full_de_lines), len(full_en_lines))

for i in range(min_len):
    de_line = full_de_lines[i]
    en_line = full_en_lines[i]
    
    if de_line and en_line:
        if de_line.count('.') <= 1 and en_line.count('.') <= 1:
            strict_full_de.append(de_line)
            strict_full_en.append(en_line)

# Filter out sentence pairs where either side starts with punctuation
filtered_pairs = [
    (de, en) for de, en in zip(strict_full_de, strict_full_en)
    if not starts_with_punctuation(de) and not starts_with_punctuation(en)
]

# Clean punctuation (keep hyphens) and strip leading/trailing spaces
filtered_de = [remove_punctuation_except_hyphen(de.strip()) for de, _ in filtered_pairs]
filtered_en = [remove_punctuation_except_hyphen(en.strip()) for _, en in filtered_pairs]

# Save the cleaned and aligned data
with open(output_de_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(filtered_de))

with open(output_en_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(filtered_en))

print(f"Processing completed, {len(filtered_de)} sentence pairs retained.")
print(f"Files saved to: {output_de_path}, {output_en_path}")
