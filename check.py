import pandas

df=pandas.read_parquet("future_regressor_dataset.parquet")
print(df.shape)
print(df.head())
print(df.columns)