import pandas as pd
import ast

# path
file_path = "E:/Language Model Project/final-model/train_15-20.csv"

# read data
df = pd.read_csv(file_path, header=None)

# Save the collection of all unique tokens
all_tokens = set()

# Iterate over each line and extract the tokens in input and output
for row in df[0]:
    try:
        input_str, output_str = row.split('][')
        input_list = ast.literal_eval(input_str + ']')
        output_list = ast.literal_eval('[' + output_str)
        all_tokens.update(input_list)
        all_tokens.update(output_list)
    except Exception as e:
        print(" Error processing row:", row, "Error:", e)

# Convert to DataFrame and save
vocab_df = pd.DataFrame(sorted(all_tokens), columns=["token"])
output_path = "E:\Language Model Project\\final-model\\Data\\train_15-20_vocab.csv"
vocab_df.to_csv(output_path, index=False)

print(f" A total of {len(all_tokens)} unique tokens were found in the training set and saved to: {output_path}")
