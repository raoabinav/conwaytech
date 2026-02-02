# Emerging Pattern Detector: Technical Design

## 1. Problem Restatement

**Goal:** Build an unsupervised detector that identifies emerging patterns in customer complaint data — issues that are new, growing, or changing in character — before they become obvious in aggregate metrics.

**Key distinction:** We're not just finding anomalies in a snapshot. We're detecting *temporal dynamics*:
- Clusters that are growing faster than baseline
- Novel failure modes appearing (weren't present historically)
- Trend changes in (company, product) pairs
- Character shifts within existing patterns

## 2. Signal Taxonomy

What types of "emerging patterns" should we detect?

| Signal Type | Description | Example | Detection Method |
|-------------|-------------|---------|------------------|
| **Volume Spike** | Sudden increase in complaints for an entity | Company X: 50 → 200 complaints/week | Statistical process control |
| **Trend Acceleration** | Gradual but accelerating growth | Company Y growing 10% → 20% → 40% week-over-week | Trend analysis |
| **Novel Cluster** | New semantic cluster appearing | "App crashes after update" emerges as new theme | Cluster novelty detection |
| **Cluster Growth** | Existing cluster growing disproportionately | "Payment processing" cluster 2x baseline | Cluster size tracking |
| **Semantic Drift** | Character of complaints changing | Same company, but issues shifting from "billing" to "harassment" | Embedding centroid drift |
| **Cross-Entity Correlation** | Multiple entities showing same pattern | 5 companies all spike on "new fee" complaints | Correlation analysis |

## 3. Design Space Exploration

### 3.1 Temporal Windowing Strategy

**Options:**

| Approach | Window Size | Pros | Cons |
|----------|-------------|------|------|
| **Fixed weekly** | 7 days | Simple, human-interpretable | Misses within-week patterns |
| **Fixed monthly** | 30 days | Smooths noise | Too coarse for fast-moving issues |
| **Rolling window** | 7d rolling, 1d step | Catches gradual changes | More complex, overlapping alerts |
| **Adaptive** | Variable based on volume | Handles low-volume entities | Complex to implement |

**Decision:** Fixed weekly windows with monthly baseline comparison.
- Weekly is granular enough to catch emerging issues
- Monthly baseline (trailing 4 weeks) provides stable comparison
- Simple to explain and debug

### 3.2 Text Embedding Strategy

**Options:**

| Approach | Model | Pros | Cons |
|----------|-------|------|------|
| **TF-IDF** | N/A | Fast, interpretable | Misses semantics, high-dim |
| **Word2Vec avg** | word2vec-google-news | Moderate quality, fast | Loses context |
| **Sentence Transformers** | all-MiniLM-L6-v2 | Good quality, reasonable speed | 384 dims, needs GPU for speed |
| **OpenAI/API** | text-embedding-3-small | High quality | Cost, latency, external dependency |

**Decision:** Sentence Transformers (all-MiniLM-L6-v2)
- Good semantic quality for clustering
- Runs locally on CPU (important for laptop constraint)
- 384 dimensions is manageable
- Well-established, reproducible

### 3.3 Clustering Strategy

**Options:**

| Approach | Algorithm | Pros | Cons |
|----------|-----------|------|------|
| **Static K-Means** | K-Means | Fast, simple | Must specify K, no temporal awareness |
| **Hierarchical** | Agglomerative | Dendrogram helps interpretation | O(n²) memory |
| **Density-based** | HDBSCAN | Finds arbitrary shapes, handles noise | Can be slow, parameter sensitive |
| **Online clustering** | River/incremental | True streaming | Complex, less mature |
| **Topic modeling** | BERTopic | Interpretable topics | May not capture emerging patterns well |

**Decision:** HDBSCAN with periodic re-clustering
- Handles noise points (many complaints are unique one-offs)
- Doesn't require specifying K
- Can compare cluster assignments across time windows
- Reasonable performance for 85k rows

### 3.4 Anomaly Detection Strategy

**Options:**

| Approach | Method | Pros | Cons |
|----------|--------|------|------|
| **Z-score** | (x - μ) / σ | Simple, interpretable | Assumes normality |
| **IQR** | 1.5 × IQR | Robust to outliers | Less sensitive |
| **Poisson test** | Exact Poisson | Appropriate for count data | Assumes constant rate |
| **Prophet/time series** | Facebook Prophet | Handles seasonality | Overkill, slow |
| **Isolation Forest** | sklearn | Multi-dimensional anomalies | Black box |

**Decision:** Poisson-based testing for volume, z-score for proportions
- Complaint counts are count data → Poisson appropriate
- Proportions (cluster share) → z-score on transformed values
- Simple, interpretable, fast

## 4. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION                           │
│  Raw CSV → Schema Mapping → Temporal Batching (weekly windows)  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FEATURE EXTRACTION                         │
│  Text → Sentence Embeddings (384d)                              │
│  Metadata → Categorical encoding                                │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PATTERN DETECTION (parallel)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Volume     │  │   Cluster    │  │   Semantic   │          │
│  │   Detector   │  │   Detector   │  │   Detector   │          │
│  │              │  │              │  │              │          │
│  │ - By company │  │ - HDBSCAN    │  │ - Novelty    │          │
│  │ - By product │  │ - Track size │  │ - Drift      │          │
│  │ - Poisson    │  │ - Growth     │  │ - Emergence  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       ALERT SYNTHESIS                           │
│  Score → Rank → Deduplicate → Contextualize → Output            │
└─────────────────────────────────────────────────────────────────┘
```

## 5. Detection Methods (Detail)

### 5.1 Volume Anomaly Detection

For each (company, week) pair:
1. Compute trailing 8-week baseline (excluding current week)
2. Model expected count as Poisson(λ = baseline_mean)
3. Compute p-value: P(X ≥ observed | λ)
4. Flag if p < 0.01 (with Bonferroni correction for multiple testing)

**Why Poisson?** Complaint counts are discrete events in time. Poisson is the natural model for "number of events in a time window."

### 5.2 Cluster-Based Detection

**Phase 1: Build global clustering**
1. Embed all complaint narratives
2. Run HDBSCAN on full dataset to identify stable clusters
3. Label each complaint with cluster ID

**Phase 2: Track cluster dynamics per window**
1. For each week, compute cluster distribution
2. Compare to trailing 4-week baseline
3. Flag clusters with:
   - Significant growth (>2σ above baseline share)
   - First appearance (new cluster emerging)
   - Significant shrinkage (potential resolution)

### 5.3 Semantic Novelty Detection

1. For each week, compute centroid of all embeddings
2. For each complaint, compute distance to historical centroid
3. Flag complaints in the top 5% of distances as "novel"
4. Group novel complaints and look for micro-clusters

**Key insight:** Novel complaints that cluster together suggest an emerging failure mode. Isolated novel complaints are just noise.

### 5.4 Cross-Entity Correlation

1. For each week, compute per-company volume deltas
2. Look for correlated spikes across multiple companies
3. When found, examine the semantic content — are they about the same issue?

**Why this matters:** A new regulation, system outage, or widespread scam will hit multiple companies simultaneously.

## 6. Alert Scoring & Ranking

Each detector produces raw signals. We need to synthesize into actionable alerts.

**Scoring dimensions:**
- **Statistical significance:** p-value or z-score
- **Magnitude:** How big is the deviation?
- **Recency:** More recent = more actionable
- **Breadth:** Single company vs. multiple entities
- **Novelty:** Is this a new pattern or recurrence?

**Composite score:**
```
score = (significance_weight × -log(p_value)) +
        (magnitude_weight × effect_size) +
        (recency_weight × recency_decay) +
        (breadth_weight × entity_count)
```

## 7. Output Format

Each alert includes:
- **Alert ID:** Unique identifier
- **Type:** volume_spike | cluster_growth | novel_cluster | semantic_drift | cross_entity
- **Severity:** high | medium | low (based on score)
- **Time window:** The week(s) where pattern detected
- **Entities involved:** Companies, products, etc.
- **Summary:** Human-readable description
- **Evidence:** Sample complaint IDs and excerpts
- **Cluster keywords:** Top terms for interpretability

## 8. Computational Considerations

**Bottlenecks:**
1. **Embedding 85k texts:** ~10-15 min on CPU with batching
2. **HDBSCAN on 85k × 384:** ~2-5 min with approximate NN
3. **Everything else:** <1 min

**Mitigation:**
- Cache embeddings to disk
- Use HDBSCAN with `core_dist_n_jobs=-1` for parallelism
- Process windows incrementally where possible

**Expected total runtime:** 15-25 minutes on laptop CPU

## 9. Validation Strategy

Without labels, validation is qualitative:

1. **Sanity checks:**
   - Do volume spikes correlate with known events (news, recalls)?
   - Are cluster labels interpretable?

2. **Manual inspection:**
   - Sample 5-10 alerts, read the underlying complaints
   - Do they represent real emerging issues?

3. **Temporal consistency:**
   - Do emerging clusters persist in subsequent weeks?
   - Or are they spurious noise?

## 10. Open Questions & Limitations

1. **Cold start:** First few weeks have no baseline — we'll exclude Oct 2023 from alerts

2. **Seasonality:** Holiday patterns may cause false positives — acknowledge but don't over-engineer

3. **Redaction noise:** 80%+ of narratives contain "XX" redaction markers — may affect embeddings slightly

4. **Single product:** All "Debt collection" — cross-product patterns not observable

5. **Channel homogeneity:** All "Web" — can't detect channel-specific patterns

## 11. Implementation Plan

| Phase | Tasks | Output |
|-------|-------|--------|
| **1. Embed** | Load data, embed narratives, cache | `embeddings.npy` |
| **2. Cluster** | HDBSCAN, label complaints | `df['cluster_id']` |
| **3. Temporal** | Window data, compute baselines | Window objects |
| **4. Volume** | Poisson tests by company/week | Volume alerts |
| **5. Cluster** | Track cluster dynamics | Cluster alerts |
| **6. Novelty** | Semantic distance analysis | Novelty alerts |
| **7. Synthesize** | Score, rank, format | Final alert report |

---

*Design document v1.0 — ready for implementation*
