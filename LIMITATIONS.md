# Limitations & Honest Assessment

## What This Is

A **proof-of-concept** detector that demonstrates:
- The methodology for detecting emerging patterns in complaint data
- How to combine clustering with statistical tests for anomaly detection
- That the approach can surface real patterns (validated against CFPB news)

## What This Is NOT

A production-ready system. Key gaps:

### 1. Not Truly Incremental

**The spec asked for:**
> "temporal batches of customer issue records"

**What that implies:**
- New batch arrives → update state → emit alerts
- Without reprocessing all historical data

**What we built:**
- Loads all data at once
- Fits models on full dataset
- Simulates batch processing but doesn't truly stream

**To fix:**
- Incremental TF-IDF (online vocabulary updates)
- Mini-batch K-Means with periodic centroid adjustment
- Persistent state store (database)

### 2. Clustering Doesn't Adapt

Problem: We fit K-Means once on all data. But in reality:
- New complaint types emerge over time
- Old patterns become irrelevant
- Vocabulary shifts

**To fix:**
- Online clustering (e.g., incremental DBSCAN)
- Periodic retraining with drift detection
- Sliding window for vocabulary updates

### 3. Single Product Only

We only tested on debt collection. Unknown if it generalizes to:
- Credit cards (different complaint patterns)
- Mortgages (longer complaint cycles)
- Multi-product datasets

### 4. TF-IDF Misses Semantics

TF-IDF treats words as independent. It misses:
- "unauthorized charge" ≈ "charge I didn't authorize" (synonyms)
- "they keep calling" ≈ "harassment" (related concepts)

**Better approach:** Sentence embeddings (e.g., sentence-transformers)
**Why we didn't:** Heavy dependencies, slow on CPU

### 5. Fixed Thresholds

We hardcoded:
- `p_threshold = 0.01` for volume spikes
- `z_threshold = 3.0` for cluster changes
- `min_company_baseline = 5`

In production: These should be tuned based on false positive feedback.

### 6. No Seasonality Handling

We don't account for:
- Holiday effects (Q4 might naturally have more complaints)
- Weekly patterns (Mondays might differ from Fridays)
- Annual cycles

This could cause false positives during predictable high-volume periods.

### 7. Alert Fatigue Risk

486 alerts over 66 weeks = 7+ alerts/week. For a human reviewer, this might be too many. Would need:
- Better deduplication
- Alert aggregation (group related alerts)
- Severity calibration

## Validation Gaps

### What We Validated

1. March 2024 credit bureau spike → confirmed against CFPB news
2. 58% of volume alerts showed sustained elevated volume

### What We Didn't Validate

1. Precision: What % of alerts are actually worth investigating?
2. Recall: What real issues did we miss?
3. Cross-product generalization
4. Performance under true streaming conditions

## Tradeoffs We Made

| Decision | Chose | Gave Up | Why |
|----------|-------|---------|-----|
| TF-IDF | Speed, interpretability | Semantic similarity | Time constraints |
| K-Means | Deterministic, fast | Flexible K, noise handling | Simpler |
| Batch processing | Simpler implementation | True streaming | Scope |
| Fixed thresholds | Quick implementation | Optimal sensitivity | No feedback loop |
| Single product | Focused demo | Generalization proof | Data availability |

## What Would Make This Production-Ready

1. **State persistence** — Database for baselines, models, alert history
2. **True incremental processing** — Process new batches without reloading
3. **Model drift detection** — Know when to retrain
4. **Feedback integration** — Learn from false positive/negative labels
5. **Seasonality modeling** — Adjust baselines for predictable patterns
6. **Alert aggregation** — Group related alerts to reduce noise
7. **Multi-product support** — Separate or shared models per product
8. **Monitoring** — Track detector health (false positive rate, latency)

## Summary

This is a **good technical demonstration** that:
- Shows the right approach
- Detects real patterns
- Has clear architecture

It is **not** a production system because:
- Not truly incremental
- Not validated at scale
- Missing operational infrastructure

A realistic next step would be to pick ONE limitation (e.g., make it truly incremental) and address it properly.
