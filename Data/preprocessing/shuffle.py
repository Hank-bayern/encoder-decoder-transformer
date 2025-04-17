import pandas as pd
import random

def shuffle_rows(input_path, output_path):
    df = pd.read_csv(input_path, header=None)
    df_shuffled = df.sample(frac=1).reset_index(drop=True)  # frac=1 means all scrambled
    df_shuffled.to_csv(output_path, index=False, header=False)
    print(f" After shuffling the lines, save them to：{output_path}")


shuffle_rows("E:\Language Model Project\\final-model\\train_15-20_10_expanded.csv", "E:\Language Model Project\\final-model\\train_15-20_10_shuffle.csv")


