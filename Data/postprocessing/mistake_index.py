import pandas as pd
import ast


file_path = "Data\\train_15-20.csv"

# read file
df = pd.read_csv(file_path, header=None)

# Extract all the tokens that appear in the second list (output) (remove duplicates)
unique_tokens = set()

for row in df[0]:
    try:
        input_str, output_str = row.split('][')
        output_list = ast.literal_eval('[' + output_str)
        unique_tokens.update(output_list)
    except Exception as e:
        print("Error processing row:", row, "Error:", e)

# save
mistake_df = pd.DataFrame(sorted(unique_tokens), columns=["token"])
output_path = "Data\\mistake_insertion_index.csv"
mistake_df.to_csv(output_path, index=False)