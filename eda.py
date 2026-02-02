import pandas as pd
import numpy as np
from datetime import datetime

# Load data
df = pd.read_csv('complaints-2026-02-01_19_29.csv', low_memory=False)

print("=" * 60)
print("BASIC STATISTICS")
print("=" * 60)
print(f"Total rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print("\n" + "=" * 60)
print("SCHEMA MAPPING TO REQUIRED COLUMNS")
print("=" * 60)
mapping = {
    'timestamp': 'Date received',
    'customer_id': 'Complaint ID',
    'channel': 'Submitted via', 
    'product': 'Product',
    'text': 'Consumer complaint narrative'
}
for req, actual in mapping.items():
    if actual in df.columns:
        null_pct = df[actual].isna().mean() * 100
        print(f"  {req} → '{actual}' (null: {null_pct:.1f}%)")

print("\n" + "=" * 60)
print("DATE RANGE")
print("=" * 60)
df['Date received'] = pd.to_datetime(df['Date received'])
print(f"Min date: {df['Date received'].min()}")
print(f"Max date: {df['Date received'].max()}")
days = (df['Date received'].max() - df['Date received'].min()).days
print(f"Span: {days} days ({days/30:.1f} months)")

print("\n" + "=" * 60)
print("MISSING VALUES (top)")
print("=" * 60)
missing = df.isna().sum().sort_values(ascending=False)
for col, count in missing.head(8).items():
    if count > 0:
        print(f"  {col}: {count:,} ({count/len(df)*100:.1f}%)")

print("\n" + "=" * 60)
print("CATEGORICAL DISTRIBUTIONS")
print("=" * 60)
for col in ['Product', 'Sub-product', 'Submitted via', 'State']:
    if col in df.columns:
        nunique = df[col].nunique()
        print(f"\n{col} ({nunique} unique):")
        print(df[col].value_counts().head(5).to_string())

print("\n" + "=" * 60)
print("COMPANY DISTRIBUTION")
print("=" * 60)
print(f"Unique companies: {df['Company'].nunique():,}")
print("\nTop 10:")
print(df['Company'].value_counts().head(10).to_string())

print("\n" + "=" * 60)
print("TEXT QUALITY (Consumer complaint narrative)")
print("=" * 60)
narratives = df['Consumer complaint narrative'].dropna()
print(f"Rows with narrative: {len(narratives):,} ({len(narratives)/len(df)*100:.1f}%)")
lengths = narratives.str.len()
print(f"Length - min: {lengths.min()}, median: {lengths.median():.0f}, max: {lengths.max()}")
print(f"Mean words: {narratives.str.split().str.len().mean():.0f}")

print("\n" + "=" * 60)
print("TEMPORAL DISTRIBUTION (complaints per month)")
print("=" * 60)
monthly = df.groupby(df['Date received'].dt.to_period('M')).size()
print(monthly.to_string())
