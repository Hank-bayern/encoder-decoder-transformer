
import pandas as pd
import ast

# 1. Read vocab (index -> token)
vocab_df = pd.read_csv("E:\Language Model Project\\final-model\\vocab_15-20.csv", header=0, names=["token"])
vocab_list = [
    str(token) if pd.notna(token) else "unk"
    for token in vocab_df["token"].tolist()
]

PAD_INDEX = 2  #  set <pad> token's index

# 2. read prediction result
df = pd.read_csv("E:\Language Model Project\\final-model\\result\\original\\100epochs\\test_result.csv", header=None)

def index_list_to_sentence(index_list, vocab):
    tokens = []
    for idx in index_list:
        try:
            idx = int(idx)
            if idx == PAD_INDEX:  # ✅ 跳过 <pad>
                continue
            token = vocab[idx] if 0 <= idx < len(vocab) else "unk"
        except:
            token = "unk"
        tokens.append(token)
    return " ".join(tokens)

# 3. translate by lines
sentences = []
for _, row in df.iterrows():
    try:
        input_indices = ast.literal_eval(row[0])
        true_indices = ast.literal_eval(row[1])
        pred_indices = ast.literal_eval(row[2])

        #  Calculate the valid token (not 0, not <pad>)
        valid_len = sum(1 for idx in true_indices if idx != 0 and idx != PAD_INDEX)
        pred_indices = pred_indices[:valid_len]  # 截断预测列表
        pred_indices = [idx for idx in pred_indices if idx != PAD_INDEX]  # 过滤 <pad>

        input_sentence = index_list_to_sentence(input_indices, vocab_list)
        true_sentence = index_list_to_sentence(true_indices, vocab_list)
        pred_sentence = index_list_to_sentence(pred_indices, vocab_list)

        sentences.append([input_sentence, true_sentence, pred_sentence])
    except Exception as e:
        print(f" Skip a line on error: {e}")
        continue

# 4. save
out_df = pd.DataFrame(sentences, columns=["input", "true_output", "pred_output"])
out_df.to_csv("E:\Language Model Project\\final-model\\result\\original\\100epochs\\translated_results.csv", index=False, encoding="utf-8")
print(" Conversion completed, output file:translated_results.csv")