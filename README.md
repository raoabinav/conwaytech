# Emerging Pattern Detector

Unsupervised detector for identifying emerging patterns in customer complaint data.

## Quick Start

```bash
# Install dependencies
pip install pandas numpy scikit-learn scipy tqdm

# Run detector
python detector.py complaints.csv

# Output: alerts.csv (ranked alerts)
```

## What It Does

Processes customer complaint data and surfaces 4 types of emerging patterns:

| Alert Type | Method | Example |
|------------|--------|---------|
| **Volume Spike** | Poisson test | Company X: 50 → 200 complaints/week |
| **Cluster Growth** | Z-test on proportions | "Fraud" theme: 2% → 10% of complaints |
| **Novel Pattern** | Distance from centroid | New complaint type not seen before |
| **Correlated Spike** | Multi-entity detection | 5 companies spike on same issue |

## Input Format

CSV with required columns:
- `Date received` — timestamp
- `Complaint ID` — unique identifier
- `Submitted via` — channel (email, web, phone)
- `Product` — product category
- `Consumer complaint narrative` — free text description

Optional: `Company`, `Sub-product`, `Issue`, `State`

## Output

`alerts.csv` with columns:
- `type` — alert type
- `week` — time window
- `severity` — high/medium/low
- `score` — ranking score
- Type-specific fields (entity, observed, expected, z_score, etc.)

## Configuration

Edit `detector.py` to adjust:
```python
n_clusters = 30          # Number of complaint clusters
baseline_weeks = 4       # Weeks for baseline comparison
p_threshold = 0.01       # Volume spike significance
z_threshold = 3.0        # Cluster growth significance
```

## Example Results

On CFPB debt collection data (85k complaints, 15 months):
- **486 alerts** detected in 39 seconds
- **Key finding:** March 2024 spike across all 3 credit bureaus (TransUnion, Equifax, Experian)

## Files

| File | Description |
|------|-------------|
| `detector.py` | Main detector implementation |
| `eda.py` | Exploratory data analysis |
| `DESIGN.md` | Architecture and tradeoffs |
| `APPROACH.md` | Implementation reasoning |
| `REPORT.md` | Results and benchmarks |
