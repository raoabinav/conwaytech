import pandas as pd
import numpy as np
from datetime import datetime

# Load data
df = pd.read_csv('complaints-2026-02-01_19_17.csv', low_memory=False)

print("=" * 60)
print("BASIC STATISTICS")
print("=" * 60)
print(f"Total rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"\nColumn names:")
for col in df.columns:
    print(f"  - {col}")

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
print(f"Span: {(df['Date received'].max() - df['Date received'].min()).days} days")

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
missing = df.isna().sum()
for col, count in missing.items():
    if count > 0:
        print(f"  {col}: {count:,} ({count/len(df)*100:.1f}%)")

print("\n" + "=" * 60)
print("CATEGORICAL DISTRIBUTIONS")
print("=" * 60)
for col in ['Product', 'Sub-product', 'Submitted via', 'State', 'Company']:
    if col in df.columns:
        nunique = df[col].nunique()
        print(f"\n{col} ({nunique} unique):")
        print(df[col].value_counts().head(5).to_string())

print("\n" + "=" * 60)
print("TEXT QUALITY (Consumer complaint narrative)")
print("=" * 60)
narratives = df['Consumer complaint narrative'].dropna()
print(f"Rows with narrative: {len(narratives):,} ({len(narratives)/len(df)*100:.1f}%)")
lengths = narratives.str.len()
print(f"Length - min: {lengths.min()}, median: {lengths.median():.0f}, max: {lengths.max()}")
print(f"Mean words: {narratives.str.split().str.len().mean():.0f}")

# Check for redaction patterns
redacted = narratives.str.contains('XX', na=False).sum()
print(f"Contains 'XX' (redaction): {redacted:,} ({redacted/len(narratives)*100:.1f}%)")

print("\n" + "=" * 60)
print("TEMPORAL DISTRIBUTION (complaints per month)")
print("=" * 60)
monthly = df.groupby(df['Date received'].dt.to_period('M')).size()
print(monthly.to_string())
