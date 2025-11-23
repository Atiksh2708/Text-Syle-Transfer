import pandas as pd

df = pd.read_parquet("blog_processed.parquet")

print(type(df.iloc[0]["sentences"]))
print(df.iloc[0]["sentences"])