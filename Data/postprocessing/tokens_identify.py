import csv
import ast

# read training dataset
file_path = "E:\Language Model Project\\final-model\Data\\train_15-20.csv"
input_tokens = set()
output_tokens = set()

with open(file_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        try:
            input_str, output_str = row[0].split('][')
            input_list = ast.literal_eval(input_str + ']')
            output_list = ast.literal_eval('[' + output_str)

            input_tokens.update(input_list)
            output_tokens.update(output_list)
        except Exception as e:
            continue

# classification
only_in_input = input_tokens - output_tokens
only_in_output = output_tokens - input_tokens
in_both = input_tokens & output_tokens



# save only_in_input
with open('E:\Language Model Project\\final-model\Data\\input_vocab.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for token in only_in_input:
        writer.writerow([token])

# save only_in_output
with open('E:\Language Model Project\\final-model\Data\\ouput_vocab.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for token in only_in_output:
        writer.writerow([token])

# save in_both
with open('E:\Language Model Project\\final-model\Data\\share_vocab.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for token in in_both:
        writer.writerow([token])
