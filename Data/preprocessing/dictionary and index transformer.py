import os
import math
import torch
import torch.nn as nn
import torchtext
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd
import csv

# path

de_input_file = "E:\Language Model Project\\final-model\Data\\filtered_de.csv"
en_input_file = "E:\Language Model Project\\final-model\Data\\filtered_en.csv"


# Read each line of text
with open(de_input_file, "r", encoding="utf-8") as de:
    de_lines = de.read().splitlines()
with open(en_input_file, "r", encoding="utf-8") as en:
    en_lines = en.read().splitlines()

# tokenizer
de_tokens = [line.strip().split() for line in de_lines]
en_tokens = [line.strip().split() for line in en_lines]




# Truncate to the same length
min_len = min(len(de_tokens), len(en_tokens))
de_tokens = de_tokens[:min_len]
en_tokens = en_tokens[:min_len]

# Filter out rows where either side has length > 50 and <40
filtered_de, filtered_en = [], []
for de, en in zip(de_tokens, en_tokens):
    if 15<=len(de) <= 20 and 15<=len(en) <= 20:
        filtered_de.append(de)
        filtered_en.append(en)





print(len(filtered_de))
#print(len(filtered_en))


#print (tokenized_lines)
tokens = (filtered_de+filtered_en)

vocab = build_vocab_from_iterator(
    tokens,
    min_freq = 1,
    specials = ["<s>","</s>","<pad>","<unk>"],
)



vocab.set_default_index(vocab["<unk>"])

print("length of vocab", len(vocab))
dict = dict((i,vocab.lookup_token(i))for i in range(len(vocab)))






# Reverse a dictionary to make a replacement
reverse_dict = {v: k for k, v in  dict.items()}  # {'eight': 0, 'five': 1, 'four': 2, ...}

# Replace data
de_indexed_data = [[reverse_dict.get(token, reverse_dict['<unk>']) for token in sentence] for sentence in filtered_de]
en_indexed_data = [[reverse_dict.get(token, reverse_dict['<unk>']) for token in sentence] for sentence in filtered_en]
#print(de_indexed_data)

# Set the maximum length
max_len = 20
pad_value = 2

de_processed_data = [row[:max_len] + [pad_value] * (max_len - len(row)) for row in de_indexed_data]
de_df = pd.DataFrame(de_processed_data)


en_processed_data = [row[:max_len] + [pad_value] * (max_len - len(row)) for row in en_indexed_data]
en_df = pd.DataFrame(en_processed_data)

en_df.to_csv("E:\Language Model Project\\final-model\Data\\en_index.csv", index=False, header=False)
de_df.to_csv("E:\Language Model Project\\final-model\Data\\de_index.csv", index=False, header=False)



print(" The word index conversion is complete and the file has been saved as de_indexed.csv")


# 保存词表为 CSV

vocab_dict = {i: vocab.lookup_token(i) for i in range(len(vocab))}
vocab_df = pd.DataFrame(list(vocab_dict.items()), columns=["index", "token"])
vocab_df.to_csv("Data\\vocab_15-20.csv", index=False, encoding="utf-8")
print(" vocabulary has benn saved as  vocab_15-20.csv")


