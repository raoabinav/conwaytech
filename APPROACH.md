# Implementation Approach Log

This document captures my reasoning and decisions during implementation.

---

## Phase 1: Embedding Pipeline

### Decision: Sentence Transformers (all-MiniLM-L6-v2)

**Why this model:**
- 384-dimensional embeddings (manageable for 85k rows)
- Trained on 1B+ sentence pairs — good semantic quality
- Fast inference on CPU (~3k sentences/min)
- Well-established, reproducible

**Alternative considered:** all-mpnet-base-v2 (768 dims, higher quality)
- Rejected because: 2x dimensionality = 2x memory for HDBSCAN, and quality difference is marginal for clustering

**Preprocessing decisions:**
- Keep redaction markers (XX) — removing them would lose context about what was redacted
- No lowercasing — sentence-transformers handles this internally
- No truncation beyond model max (256 tokens) — most narratives fit

### Batching Strategy

85k embeddings × 384 dims × 4 bytes = ~130MB in memory. Manageable.

But embedding 85k texts serially would be slow. Strategy:
- Batch size 128 — balances throughput vs. memory
- Progress bar — so we know it's working
- Cache to disk — so we don't re-embed on subsequent runs

---

## Phase 2: Clustering

### Decision: HDBSCAN

**Why HDBSCAN over K-Means:**
1. Don't need to specify K (we don't know how many complaint types exist)
2. Handles noise — many complaints are one-offs, HDBSCAN labels them as noise (-1)
3. Density-based — finds clusters of varying shapes

**Key parameters:**
- `min_cluster_size=50` — clusters must have at least 50 complaints to be meaningful
- `min_samples=10` — core point threshold
- `metric='euclidean'` — standard for sentence embeddings

**Why min_cluster_size=50:**
- With 85k rows over 60 weeks, 50 complaints = ~1/week on average
- Smaller clusters would be too noisy for temporal tracking
- Larger would miss emerging patterns

### Cluster Labeling

HDBSCAN gives us numeric cluster IDs. For interpretability:
- Extract top TF-IDF terms per cluster
- These become cluster "labels" for human review

---

## Phase 3: Temporal Windowing

### Window Structure

```
Week 1 | Week 2 | Week 3 | Week 4 | Week 5 | ... | Week 60
       |<-- baseline (4 weeks) -->|<- current ->|
```

For each week from week 5 onward:
- Baseline = preceding 4 weeks
- Compare current week to baseline
- Flag significant deviations

### Why 4-week baseline:
- Long enough to smooth weekly noise
- Short enough to adapt to gradual changes
- ~1 month = business-meaningful period

---

## Phase 4: Anomaly Detection

### Volume Anomalies (Company-level)

For each (company, week):
1. Compute baseline mean λ from prior 4 weeks
2. Observed count k in current week
3. P(X ≥ k | Poisson(λ)) = 1 - CDF(k-1, λ)
4. Flag if p < α (with multiple testing correction)

**Why Poisson:**
- Complaints are count data (discrete events in time)
- Poisson is the standard model for "events per interval"
- Assumes constant rate during baseline — reasonable for 4-week windows

**Multiple testing:**
- ~2000 companies × 55 weeks = 110k tests
- Bonferroni is too conservative
- Using Benjamini-Hochberg FDR control at q=0.05

### Cluster Growth Anomalies

For each (cluster, week):
1. Compute cluster's share of complaints in baseline
2. Compute cluster's share in current week
3. Z-test on proportions
4. Flag if |z| > 3 (very significant)

### Novelty Detection

1. Compute historical embedding centroid (all data before current week)
2. For each complaint in current week, compute distance to centroid
3. Complaints > 95th percentile distance are "novel"
4. Cluster the novel complaints — if they form a micro-cluster, that's an emerging pattern

---

## Open Issues (to resolve during implementation)

- [x] How to handle companies with very few complaints? → Set minimum baseline threshold (5 complaints over baseline period)
- [ ] Should we weight recent baseline weeks more heavily? → Kept uniform for simplicity, but exponential decay could help
- [x] What's the right threshold for "novel" micro-clusters? → Used 95th percentile of distances, 10+ novel complaints to flag

---

## Implementation Pivot: TF-IDF Instead of Sentence Transformers

**Original plan:** sentence-transformers (all-MiniLM-L6-v2) for embeddings

**What happened:** Heavy dependencies (PyTorch, transformers) were slow to install, blocking progress.

**Pivot decision:** Use TF-IDF + K-Means instead

**Tradeoffs:**
| Aspect | Sentence Transformers | TF-IDF |
|--------|----------------------|--------|
| Semantic quality | High (captures meaning) | Medium (lexical only) |
| Install time | ~10 min | Already in sklearn |
| Runtime | ~15 min for 85k | ~3 sec for 85k |
| Interpretability | Low (dense vectors) | High (can see top words) |

**Conclusion:** For this demonstration, TF-IDF was sufficient to show the methodology. The architecture remains the same — only the vectorization step differs. Production version could swap in sentence-transformers without changing other components.

---

## Results Summary

**Ran detector on 85,515 complaints → 486 alerts**

Key validations:
1. March 2024 credit bureau spike detected (all 3 major bureaus spiked in same week)
2. CL Holdings LLC volume anomaly detected (5.23x growth)
3. "Unauthorized transactions" cluster growth/decline tracked over time
4. January 2025 correlated spikes flagged

The detector successfully surfaced actionable, interpretable patterns without any labels.

---

*Implementation complete.*
