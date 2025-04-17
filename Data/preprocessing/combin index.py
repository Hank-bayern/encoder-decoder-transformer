import pandas as pd


# read files
de_file_path = "E:\Language Model Project\\final-model\Data\\de_index.csv"  
en_file_path = "E:\Language Model Project\\final-model\Data\\en_index.csv"  

df_de = pd.read_csv(de_file_path,  header=None)
df_en = pd.read_csv(en_file_path,  header=None)

# Make sure the two files have the same number of lines.
min_length = min(len(df_de), len(df_en))
df_de = df_de.iloc[:min_length]
df_en = df_en.iloc[:min_length]

# Merge lines into list format, forming a ([de list][en list]) structure
df_combined = pd.DataFrame({
    "combined": df_en.apply(lambda x: list(x), axis=1).astype(str) + df_de.apply(lambda x: list(x), axis=1).astype(str)
})

# save
combined_file_path = "E:\Language Model Project\\final-model\Data\\combined_index.csv"
df_combined.to_csv(combined_file_path, index=False, header=False)

print(f"saved as: {combined_file_path}")