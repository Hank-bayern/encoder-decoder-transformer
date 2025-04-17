import pandas as pd


input_file = "E:/Language Model Project/final-model/result/expanded 10 times/20epochs/translation_mistake_result.csv"
output_file = "E:/Language Model Project/final-model/result/expanded 10 times/20epochs/translation_mistake_accuracy.csv"

input_vocab_file = "E:/Language Model Project/final-model/Data/input_vocab.csv"
output_vocab_file = "E:/Language Model Project/final-model/Data/ouput_vocab.csv"
share_vocab_file = "E:/Language Model Project/final-model/Data/share_vocab.csv"

pad_token_id = 2  # pad 的 token index



# === Step 1: read data ===
df = pd.read_csv(input_file, header=None)
df = df.iloc[1:].reset_index(drop=True)
df[1] = df[1].apply(eval)  # true_output
df[2] = df[2].apply(eval)  # pred_output
'''
df['trimmed_true'] = df[1]
df['trimmed_pred'] = df[2]

'''
df['trimmed_true'] = df[1].apply(lambda x: x[1:])# remove first token of true_output
df['trimmed_pred'] = df[2].apply(lambda x: x[1:])# remove first token of pred_output 


# === Step 2: load vocab list ===
input_vocab = set(pd.read_csv(input_vocab_file, header=None)[0].astype(int).tolist())
output_vocab = set(pd.read_csv(output_vocab_file, header=None)[0].astype(int).tolist())
share_vocab = set(pd.read_csv(share_vocab_file, header=None)[0].astype(int).tolist())
valid_token_set = input_vocab | output_vocab | share_vocab

# === Step 3: Number of valid tokens (not pad)===
df['valid_tokens'] = df['trimmed_true'].apply(lambda x: sum(t != 2 for t in x))

# === Step 4: Count each type of error ===
def classify_errors(row):
    true = row['trimmed_true']
    pred = row['trimmed_pred']
    valid_len = row['valid_tokens']

    correct = 0
    type1, type2, type3 = 0, 0, 0

    for t, p in zip(true[:valid_len], pred[:valid_len]):
        if p == t:
            correct += 1
        elif p not in valid_token_set:
            type3 += 1
        elif p in input_vocab:
            type1 += 1
        else:
            type2 += 1

    return pd.Series([correct, type1, type2, type3])

df[['correct', 'type1_input_error', 'type2_output_share_error', 'type3_ood_error']] = df.apply(classify_errors, axis=1)

# === Step 5: Summary & Calculation of Ratios ===
df['total'] = df['valid_tokens']
df['accuracy'] = df['correct'] / df['total']
df['type1_ratio'] = df['type1_input_error'] / df['total']
df['type2_ratio'] = df['type2_output_share_error'] / df['total']
df['type3_ratio'] = df['type3_ood_error'] / df['total']
df['hallucniation'] = df['type1_input_error'] + df['type3_ood_error']
df['hallucniation_ratio'] = df['type1_ratio'] + df['type3_ratio']
df['translation_mistake'] = df['type2_output_share_error']
df['translation_mistake_ratio'] = df['type2_ratio']


# === Step 6:  Overall Statistics ===
total = df['total'].sum()
total_correct = df['correct'].sum()
total_hallucination = df['type1_input_error'].sum()+df['type3_ood_error'].sum()
total_translation_mistake = df['type2_output_share_error'].sum()


print(f" Accuracy: {total_correct / total:.4f}")
print(f" Translation Mistake: {total_translation_mistake} ({total_translation_mistake / total:.4%})")
print(f" Hallucniation: {total_hallucination} ({total_hallucination / total:.4%})")



# === Step 7: save ===
df[['correct', 'total', 'accuracy', 'translation_mistake', 'translation_mistake_ratio', 'hallucniation',
    'hallucniation_ratio']].to_csv(output_file, index=False)
print(f" saved as: {output_file}")



