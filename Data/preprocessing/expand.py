import pandas as pd

# load file
original_path = "E:\Language Model Project\\final-model\\train_15-20.csv"
df = pd.read_csv(original_path)

# copy 10 times
expanded_df = pd.concat([df] * 10, ignore_index=True)

# save
expanded_path = "E:\Language Model Project\\final-model\\train_15-20_10_expanded.csv"
expanded_df.to_csv(expanded_path, index=False)


expanded_path