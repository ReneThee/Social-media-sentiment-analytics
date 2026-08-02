"""Looking at the raw dataset"""
import pandas as pd

df = pd.read_csv("data/sentimentdataset.csv")

print("Shape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nDtypes:\n", df.dtypes)
print("\nFirst 3 rows:\n", df.head(3))

# The three columns everything depends on — inspect their actual contents
for col in df.columns:
    if col.strip().lower() in ("platform", "sentiment"):
        print(f"\nUnique values in {col!r} ({df[col].nunique()}):")
        print(df[col].unique()[:40])

# Missing values 
print("\nMissing values per column:\n", df.isna().sum())