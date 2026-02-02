"""
Emerging Pattern Detector for Customer Complaints

Unsupervised detector that identifies:
1. Volume spikes (company-level complaint surges)
2. Cluster growth (semantic themes gaining traction)
3. Novel patterns (new complaint types emerging)
4. Cross-entity correlations (systemic issues)

Approach: TF-IDF + K-Means clustering + statistical anomaly detection
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from scipy import stats
from collections import defaultdict
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

def load_data(path: str) -> pd.DataFrame:
    """Load and preprocess complaint data."""
    df = pd.read_csv(path, low_memory=False)

    # Rename to standard schema
    df = df.rename(columns={
        'Date received': 'timestamp',
        'Complaint ID': 'complaint_id',
        'Submitted via': 'channel',
        'Product': 'product',
        'Consumer complaint narrative': 'text',
        'Company': 'company',
        'Sub-product': 'sub_product',
        'Issue': 'issue',
        'State': 'state'
    })

    # Parse dates
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Create week identifier (ISO week)
    df['week'] = df['timestamp'].dt.to_period('W')

    # Clean text - basic preprocessing
    df['text_clean'] = df['text'].fillna('').str.lower()

    print(f"Loaded {len(df):,} complaints from {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    print(f"Unique companies: {df['company'].nunique():,}")
    print(f"Unique weeks: {df['week'].nunique()}")

    return df


# =============================================================================
# TEXT VECTORIZATION (TF-IDF)
# =============================================================================

def build_tfidf_features(df: pd.DataFrame, max_features: int = 5000) -> tuple:
    """
    Build TF-IDF features from complaint text.

    Why TF-IDF:
    - Fast, no external dependencies
    - Interpretable (we can see which words matter)
    - Good enough for clustering similar complaints

    Trade-off vs embeddings:
    - Misses synonyms and semantic similarity
    - But much faster and still captures lexical patterns
    """
    print("\nBuilding TF-IDF features...")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),  # unigrams and bigrams
        min_df=5,  # must appear in at least 5 docs
        max_df=0.8  # ignore terms in >80% of docs
    )

    tfidf_matrix = vectorizer.fit_transform(df['text_clean'])

    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    return tfidf_matrix, vectorizer


# =============================================================================
# CLUSTERING
# =============================================================================

def cluster_complaints(tfidf_matrix, n_clusters: int = 30) -> np.ndarray:
    """
    Cluster complaints using MiniBatch K-Means.

    Why K-Means over HDBSCAN:
    - Much faster (O(n) vs O(n²))
    - No heavy dependencies
    - For this dataset size, works well

    Why n_clusters=30:
    - With 85k complaints, want meaningful groupings
    - Too few = overly broad clusters
    - Too many = hard to track temporally
    - 30 gives ~2800 complaints/cluster on average
    """
    print(f"\nClustering into {n_clusters} clusters...")

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=1000,
        n_init=3
    )

    labels = kmeans.fit_predict(tfidf_matrix)

    # Print cluster distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Cluster sizes - min: {counts.min()}, max: {counts.max()}, median: {np.median(counts):.0f}")

    return labels, kmeans


def get_cluster_labels(vectorizer, kmeans, n_terms: int = 5) -> dict:
    """Extract top terms for each cluster for interpretability."""
    terms = vectorizer.get_feature_names_out()
    cluster_labels = {}

    for i, center in enumerate(kmeans.cluster_centers_):
        top_indices = center.argsort()[-n_terms:][::-1]
        top_terms = [terms[idx] for idx in top_indices]
        cluster_labels[i] = ', '.join(top_terms)

    return cluster_labels


# =============================================================================
# TEMPORAL WINDOWING
# =============================================================================

def create_weekly_windows(df: pd.DataFrame, baseline_weeks: int = 4):
    """
    Create weekly windows with rolling baseline.

    Structure:
    - For each week, compute stats
    - Baseline = preceding N weeks
    - Detect anomalies relative to baseline
    """
    weeks = sorted(df['week'].unique())

    print(f"\nCreating {len(weeks)} weekly windows with {baseline_weeks}-week baseline")

    windows = []
    for i, week in enumerate(weeks):
        window = {
            'week': week,
            'week_idx': i,
            'data': df[df['week'] == week],
            'baseline_weeks': weeks[max(0, i-baseline_weeks):i] if i > 0 else [],
            'has_baseline': i >= baseline_weeks
        }
        windows.append(window)

    return windows


# =============================================================================
# VOLUME ANOMALY DETECTION
# =============================================================================

def detect_volume_anomalies(df: pd.DataFrame, windows: list,
                            min_baseline_count: int = 5,
                            p_threshold: float = 0.01) -> list:
    """
    Detect volume spikes for companies using Poisson test.

    Method:
    1. For each company, compute baseline rate (complaints/week)
    2. Model as Poisson(λ = baseline_mean)
    3. Test if current week's count is surprisingly high

    Why Poisson:
    - Complaints are count data (discrete events per time window)
    - Poisson is the natural model for this
    - More appropriate than z-score for low-count data
    """
    print("\nDetecting volume anomalies...")

    alerts = []

    for window in windows:
        if not window['has_baseline']:
            continue

        week = window['week']
        current_data = window['data']

        # Get baseline data
        baseline_data = df[df['week'].isin(window['baseline_weeks'])]

        # Current week counts by company
        current_counts = current_data.groupby('company').size()

        # Baseline weekly average by company
        baseline_counts = baseline_data.groupby(['company', 'week']).size().reset_index(name='count')
        baseline_avg = baseline_counts.groupby('company')['count'].mean()

        # Test each company
        for company in current_counts.index:
            observed = current_counts[company]
            expected = baseline_avg.get(company, 0)

            # Skip if baseline too low (high variance)
            if expected < min_baseline_count / len(window['baseline_weeks']):
                continue

            # Poisson test: P(X >= observed | lambda=expected)
            # Using survival function (1 - CDF)
            p_value = 1 - stats.poisson.cdf(observed - 1, expected)

            if p_value < p_threshold:
                # Calculate effect size
                ratio = observed / expected if expected > 0 else float('inf')

                alerts.append({
                    'type': 'volume_spike',
                    'week': str(week),
                    'entity': company,
                    'observed': int(observed),
                    'expected': round(expected, 1),
                    'ratio': round(ratio, 2),
                    'p_value': round(p_value, 6),
                    'severity': 'high' if ratio > 3 else 'medium' if ratio > 2 else 'low'
                })

    print(f"Found {len(alerts)} volume anomalies")
    return alerts


# =============================================================================
# CLUSTER DYNAMICS DETECTION
# =============================================================================

def detect_cluster_anomalies(df: pd.DataFrame, windows: list,
                              z_threshold: float = 3.0) -> list:
    """
    Detect clusters growing/shrinking faster than expected.

    Method:
    1. For each cluster, track share of complaints over time
    2. Compare current week's share to baseline
    3. Z-test on proportions

    Why z-test on proportions:
    - Cluster share is a proportion (0-1)
    - Z-test appropriate for proportion comparisons
    - z > 3 means very unlikely by chance
    """
    print("\nDetecting cluster anomalies...")

    alerts = []

    for window in windows:
        if not window['has_baseline']:
            continue

        week = window['week']
        current_data = window['data']
        baseline_data = df[df['week'].isin(window['baseline_weeks'])]

        n_current = len(current_data)
        n_baseline = len(baseline_data)

        if n_current < 10 or n_baseline < 10:
            continue

        # Cluster distributions
        current_dist = current_data['cluster'].value_counts(normalize=True)
        baseline_dist = baseline_data['cluster'].value_counts(normalize=True)

        for cluster_id in current_dist.index:
            p_current = current_dist[cluster_id]
            p_baseline = baseline_dist.get(cluster_id, 0)

            if p_baseline < 0.01:  # Skip very rare clusters
                continue

            # Z-test for difference in proportions
            p_pooled = (p_current * n_current + p_baseline * n_baseline) / (n_current + n_baseline)
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_current + 1/n_baseline))

            if se > 0:
                z = (p_current - p_baseline) / se
            else:
                continue

            if abs(z) > z_threshold:
                direction = 'growing' if z > 0 else 'shrinking'

                alerts.append({
                    'type': f'cluster_{direction}',
                    'week': str(week),
                    'cluster_id': int(cluster_id),
                    'current_share': round(p_current * 100, 2),
                    'baseline_share': round(p_baseline * 100, 2),
                    'z_score': round(z, 2),
                    'severity': 'high' if abs(z) > 5 else 'medium'
                })

    print(f"Found {len(alerts)} cluster anomalies")
    return alerts


# =============================================================================
# NOVELTY DETECTION
# =============================================================================

def detect_novelty(df: pd.DataFrame, tfidf_matrix, windows: list,
                   percentile_threshold: float = 95) -> list:
    """
    Detect novel complaints that don't fit historical patterns.

    Method:
    1. For each week, compute distance from each complaint to historical centroid
    2. Flag complaints in top 5% of distances
    3. Look for micro-clusters among novel complaints

    Key insight:
    - Isolated novel complaints = noise
    - Clustered novel complaints = emerging pattern
    """
    print("\nDetecting novelty...")

    alerts = []

    # Convert sparse to dense for centroid computation
    # (only do this for small chunks to manage memory)

    for window in windows:
        if not window['has_baseline']:
            continue

        week = window['week']
        current_idx = window['data'].index.tolist()
        baseline_data = df[df['week'].isin(window['baseline_weeks'])]
        baseline_idx = baseline_data.index.tolist()

        if len(current_idx) < 10 or len(baseline_idx) < 50:
            continue

        # Compute baseline centroid
        baseline_vectors = tfidf_matrix[baseline_idx].toarray()
        centroid = baseline_vectors.mean(axis=0)

        # Compute distances for current week
        current_vectors = tfidf_matrix[current_idx].toarray()
        distances = np.linalg.norm(current_vectors - centroid, axis=1)

        # Find outliers (top 5% by distance)
        threshold = np.percentile(distances, percentile_threshold)
        novel_mask = distances > threshold
        novel_count = novel_mask.sum()

        if novel_count > 10:  # Meaningful number of novel complaints
            # Get sample complaint IDs
            novel_indices = np.array(current_idx)[novel_mask][:5]
            sample_ids = df.loc[novel_indices, 'complaint_id'].tolist()

            alerts.append({
                'type': 'novel_cluster',
                'week': str(week),
                'novel_count': int(novel_count),
                'total_count': len(current_idx),
                'novel_share': round(novel_count / len(current_idx) * 100, 2),
                'sample_complaint_ids': sample_ids,
                'severity': 'medium'
            })

    print(f"Found {len(alerts)} novelty alerts")
    return alerts


# =============================================================================
# CROSS-ENTITY CORRELATION
# =============================================================================

def detect_correlated_spikes(df: pd.DataFrame, windows: list,
                             min_companies: int = 3,
                             correlation_threshold: float = 0.7) -> list:
    """
    Detect when multiple companies spike together (systemic issue).

    Method:
    1. For each week, identify companies with significant volume increases
    2. If many companies spike together, flag as systemic
    3. Look at semantic content to understand what's happening
    """
    print("\nDetecting correlated spikes...")

    alerts = []

    for window in windows:
        if not window['has_baseline']:
            continue

        week = window['week']
        current_data = window['data']
        baseline_data = df[df['week'].isin(window['baseline_weeks'])]

        # Compute growth ratios by company
        current_counts = current_data.groupby('company').size()
        baseline_avg = baseline_data.groupby('company').size() / len(window['baseline_weeks'])

        # Find companies with >2x growth
        growth_ratios = {}
        for company in current_counts.index:
            if company in baseline_avg.index and baseline_avg[company] >= 2:
                ratio = current_counts[company] / baseline_avg[company]
                if ratio > 2:
                    growth_ratios[company] = ratio

        if len(growth_ratios) >= min_companies:
            # Check if these companies share similar complaint content
            spiking_companies = list(growth_ratios.keys())
            spiking_data = current_data[current_data['company'].isin(spiking_companies)]

            # Get top issues among spiking companies
            top_issues = spiking_data['issue'].value_counts().head(3).to_dict()

            alerts.append({
                'type': 'correlated_spike',
                'week': str(week),
                'companies_affected': len(growth_ratios),
                'company_list': spiking_companies[:10],  # Top 10
                'avg_growth_ratio': round(np.mean(list(growth_ratios.values())), 2),
                'common_issues': top_issues,
                'severity': 'high' if len(growth_ratios) >= 5 else 'medium'
            })

    print(f"Found {len(alerts)} correlated spike alerts")
    return alerts


# =============================================================================
# ALERT SYNTHESIS & RANKING
# =============================================================================

def synthesize_alerts(volume_alerts: list, cluster_alerts: list,
                      novelty_alerts: list, correlation_alerts: list,
                      cluster_labels: dict) -> pd.DataFrame:
    """
    Combine all alerts, add context, and rank by importance.
    """
    all_alerts = []

    # Add cluster labels to cluster alerts
    for alert in cluster_alerts:
        alert['cluster_label'] = cluster_labels.get(alert['cluster_id'], 'unknown')
        all_alerts.append(alert)

    all_alerts.extend(volume_alerts)
    all_alerts.extend(novelty_alerts)
    all_alerts.extend(correlation_alerts)

    if not all_alerts:
        return pd.DataFrame()

    df_alerts = pd.DataFrame(all_alerts)

    # Score for ranking
    def compute_score(row):
        score = 0

        # Severity
        if row.get('severity') == 'high':
            score += 10
        elif row.get('severity') == 'medium':
            score += 5

        # Statistical significance
        if 'p_value' in row and pd.notna(row['p_value']):
            score += -np.log10(row['p_value'] + 1e-10)
        if 'z_score' in row and pd.notna(row['z_score']):
            score += abs(row['z_score'])

        # Magnitude
        if 'ratio' in row and pd.notna(row['ratio']):
            score += min(row['ratio'], 10)  # Cap at 10

        return score

    df_alerts['score'] = df_alerts.apply(compute_score, axis=1)
    df_alerts = df_alerts.sort_values('score', ascending=False)

    return df_alerts


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_detector(data_path: str, output_path: str = 'alerts.csv'):
    """
    Run the complete emerging pattern detector pipeline.
    """
    print("=" * 60)
    print("EMERGING PATTERN DETECTOR")
    print("=" * 60)

    # 1. Load data
    df = load_data(data_path)

    # 2. Build TF-IDF features
    tfidf_matrix, vectorizer = build_tfidf_features(df)

    # 3. Cluster complaints
    labels, kmeans = cluster_complaints(tfidf_matrix, n_clusters=30)
    df['cluster'] = labels
    cluster_labels = get_cluster_labels(vectorizer, kmeans)

    print("\nCluster labels (top terms):")
    for i, label in list(cluster_labels.items())[:10]:
        print(f"  Cluster {i}: {label}")

    # 4. Create temporal windows
    windows = create_weekly_windows(df, baseline_weeks=4)

    # 5. Run detectors
    volume_alerts = detect_volume_anomalies(df, windows)
    cluster_alerts = detect_cluster_anomalies(df, windows)
    novelty_alerts = detect_novelty(df, tfidf_matrix, windows)
    correlation_alerts = detect_correlated_spikes(df, windows)

    # 6. Synthesize and rank
    df_alerts = synthesize_alerts(
        volume_alerts, cluster_alerts, novelty_alerts, correlation_alerts,
        cluster_labels
    )

    print("\n" + "=" * 60)
    print(f"TOTAL ALERTS: {len(df_alerts)}")
    print("=" * 60)

    if len(df_alerts) > 0:
        print("\nAlert breakdown:")
        print(df_alerts['type'].value_counts().to_string())

        print(f"\nSeverity breakdown:")
        print(df_alerts['severity'].value_counts().to_string())

        # Save to CSV
        df_alerts.to_csv(output_path, index=False)
        print(f"\nAlerts saved to {output_path}")

        # Print top 20 alerts
        print("\n" + "=" * 60)
        print("TOP 20 ALERTS")
        print("=" * 60)

        for i, row in df_alerts.head(20).iterrows():
            print(f"\n[{row['severity'].upper()}] {row['type']} - Week {row['week']}")
            if row['type'] == 'volume_spike':
                print(f"  Company: {row['entity']}")
                print(f"  Observed: {row['observed']} vs Expected: {row['expected']} ({row['ratio']}x)")
            elif 'cluster' in row['type']:
                print(f"  Cluster: {row.get('cluster_label', 'N/A')}")
                print(f"  Share: {row['baseline_share']}% → {row['current_share']}% (z={row['z_score']})")
            elif row['type'] == 'novel_cluster':
                print(f"  Novel complaints: {row['novel_count']} ({row['novel_share']}% of week)")
            elif row['type'] == 'correlated_spike':
                print(f"  Companies affected: {row['companies_affected']}")
                print(f"  Avg growth: {row['avg_growth_ratio']}x")

    return df, df_alerts, cluster_labels


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else 'complaints-2026-02-01_19_29.csv'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'alerts.csv'

    df, alerts, cluster_labels = run_detector(data_path, output_path)
