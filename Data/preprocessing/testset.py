import pandas as pd
from sklearn.model_selection import train_test_split

# files
file_path = "E:\Language Model Project\\final-model\Data\preprocessing\\combined_index.csv"

# read data
df = pd.read_csv(file_path, header=None)

# Step 1: Divide the training set and test set (8:2)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Step 2: Construct the vocab of the training set (i.e. the set of tokens that have appeared)
train_tokens = set()
for row in train_df[0]:
    tokens = [int(tok) for tok in row.strip('[]').split(',') if tok.strip().isdigit()]
    train_tokens.update(tokens)

# Step 3: Filter the rows in the test set that contain OOV (not in the training set vocab) tokens
def is_valid_row(row):
    tokens = [int(tok) for tok in row.strip('[]').split(',') if tok.strip().isdigit()]
    return all(tok in train_tokens for tok in tokens)

filtered_test_df = test_df[test_df[0].apply(is_valid_row)].reset_index(drop=True)

# Step 4: Save the processed data
train_path = "E:\Language Model Project\\final-model\Data\preprocessing\\train_15-20.csv"
test_path = "E:\Language Model Project\\final-model\Data\preprocessing\\test_15-20.csv"

train_df.to_csv(train_path, index=False, header=False)
filtered_test_df.to_csv(test_path, index=False, header=False)

print(f"saved as:\ntrain: {train_path}\ntest: {test_path}")
