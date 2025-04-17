
import pandas as pd
import csv
import ast
import random


output_index_path = "Data\mistake_inserion_index.csv"
test_data_path = "Data\\test_15-20.csv"
translation_mistake_path = "Data\\translation_mistake_insertion_15-20.csv"

# Read all tokens in mistake_insertion_index.csv
output_tokens_df = pd.read_csv(output_index_path)
output_token_list = output_tokens_df['token'].dropna().astype(int).tolist()

# Modify the test set
modified_rows = []

with open(test_data_path, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        try:
            input_str, output_str = row[0].split('][')
            input_list = ast.literal_eval(input_str + ']')
            output_list = ast.literal_eval('[' + output_str)

            # Randomly select a token for each line and insert it at the front
            token = random.choice(output_token_list)
            input_list = [token] + input_list
            output_list = [token] + output_list

            modified_rows.append([str(input_list) + str(output_list)])
        except Exception as e:
            continue  # Ignore lines that failed to parse

# Save the new test set
with open(translation_mistake_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(modified_rows)

translation_mistake_path
