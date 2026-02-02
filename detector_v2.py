"""
Emerging Pattern Detector for Customer Complaints
==================================================

A stateful, batch-oriented detector that identifies emerging patterns
in customer complaint data.

Architecture:
- DetectorState: Maintains historical baselines, cluster models, alert history
- Detectors: Modular detection components (volume, cluster, novelty)
- Orchestrator: Processes batches, updates state, emits alerts

Design Decisions:
- Batch-oriented (not true streaming) but maintains state between batches
- Weekly granularity balances signal vs noise
- 4-week rolling baseline for "normal" definition
- Multiple detection types run in parallel

Tradeoffs Made:
- TF-IDF over embeddings: faster, interpretable, but misses semantics
- K-Means over HDBSCAN: fixed K but much faster
- Poisson over z-score for counts: more appropriate for discrete events
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from scipy import stats
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Alert:
    """Single alert emitted by the detector."""
    alert_id: str
    alert_type: str  # volume_spike, cluster_growth, cluster_shrinking, novel_pattern, correlated_spike
    severity: str    # high, medium, low
    week: str
    description: str
    evidence: Dict
    score: float

    def to_dict(self) -> Dict:
        return {
            'alert_id': self.alert_id,
            'type': self.alert_type,
            'severity': self.severity,
            'week': self.week,
            'description': self.description,
            'score': self.score,
            **self.evidence
        }


@dataclass
class DetectorState:
    """
    Maintains state between batch processing runs.

    In a production system, this would be persisted to a database.
    Here we keep it in memory but the structure shows what needs to persist.
    """
    # Text processing models (would be serialized/loaded in production)
    vectorizer: Optional[TfidfVectorizer] = None
    cluster_model: Optional[MiniBatchKMeans] = None
    cluster_labels: Dict[int, str] = field(default_factory=dict)

    # Historical data for baseline computation
    # Key: week string, Value: DataFrame of that week's data
    weekly_data: Dict[str, pd.DataFrame] = field(default_factory=dict)

    # Processed statistics per week
    # Key: week, Value: {company: count, cluster_dist: {...}, ...}
    weekly_stats: Dict[str, Dict] = field(default_factory=dict)

    # Alert history (to avoid duplicate alerts)
    emitted_alerts: set = field(default_factory=set)

    # Configuration
    baseline_weeks: int = 4
    min_company_baseline: int = 5  # minimum complaints in baseline to flag

    def get_baseline_weeks(self, current_week: str) -> List[str]:
        """Get the N weeks before current_week for baseline."""
        all_weeks = sorted(self.weekly_data.keys())
        if current_week not in all_weeks:
            return []
        idx = all_weeks.index(current_week)
        start = max(0, idx - self.baseline_weeks)
        return all_weeks[start:idx]

    def has_sufficient_history(self, current_week: str) -> bool:
        """Check if we have enough history for meaningful detection."""
        return len(self.get_baseline_weeks(current_week)) >= self.baseline_weeks


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

class FeatureExtractor:
    """
    Handles text vectorization and clustering.

    Design Decision: TF-IDF + K-Means
    - TF-IDF: Fast, interpretable, no external dependencies
    - K-Means: Deterministic, fast, well-understood

    Tradeoff: Loses semantic similarity (synonyms treated as different)
    Alternative: Sentence-transformers would capture "unauthorized charge" ≈ "charge I didn't make"
    """

    def __init__(self, max_features: int = 5000, n_clusters: int = 30):
        self.max_features = max_features
        self.n_clusters = n_clusters

    def fit(self, texts: pd.Series, state: DetectorState) -> np.ndarray:
        """
        Fit vectorizer and clustering on initial data.

        Called once during initialization or when retraining is needed.
        Returns cluster assignments for the input texts.
        """
        # TF-IDF vectorization
        state.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.8
        )
        tfidf_matrix = state.vectorizer.fit_transform(texts.fillna('').str.lower())

        # Clustering
        state.cluster_model = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            batch_size=1000,
            n_init=3
        )
        labels = state.cluster_model.fit_predict(tfidf_matrix)

        # Extract cluster labels (top terms per cluster)
        terms = state.vectorizer.get_feature_names_out()
        for i, center in enumerate(state.cluster_model.cluster_centers_):
            top_indices = center.argsort()[-5:][::-1]
            top_terms = [terms[idx] for idx in top_indices]
            state.cluster_labels[i] = ', '.join(top_terms)

        return labels

    def transform(self, texts: pd.Series, state: DetectorState) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform new texts using fitted models.

        Returns (tfidf_matrix, cluster_labels)
        """
        if state.vectorizer is None or state.cluster_model is None:
            raise ValueError("Must call fit() before transform()")

        tfidf_matrix = state.vectorizer.transform(texts.fillna('').str.lower())
        labels = state.cluster_model.predict(tfidf_matrix)
        return tfidf_matrix, labels


# =============================================================================
# DETECTORS
# =============================================================================

class VolumeDetector:
    """
    Detects volume spikes for entities (companies).

    Method: Poisson test
    - Model expected count as Poisson(λ = baseline_mean)
    - Flag if P(X >= observed) < threshold

    Why Poisson: Complaints are discrete events in time.
    Alternative: Z-score assumes normality, less appropriate for counts.
    """

    def __init__(self, p_threshold: float = 0.01):
        self.p_threshold = p_threshold

    def detect(self, current_week: str, state: DetectorState) -> List[Alert]:
        if not state.has_sufficient_history(current_week):
            return []

        alerts = []
        current_data = state.weekly_data[current_week]
        baseline_weeks = state.get_baseline_weeks(current_week)

        # Current counts by company
        current_counts = current_data.groupby('company').size()

        # Baseline average by company
        baseline_data = pd.concat([state.weekly_data[w] for w in baseline_weeks])
        baseline_counts = baseline_data.groupby('company').size() / len(baseline_weeks)

        for company in current_counts.index:
            observed = current_counts[company]
            expected = baseline_counts.get(company, 0)

            # Skip low-volume companies (high variance, noisy)
            if expected < state.min_company_baseline / len(baseline_weeks):
                continue

            # Poisson test: P(X >= observed | lambda = expected)
            p_value = 1 - stats.poisson.cdf(observed - 1, expected)

            if p_value < self.p_threshold:
                ratio = observed / expected if expected > 0 else float('inf')
                severity = 'high' if ratio > 3 else 'medium' if ratio > 2 else 'low'

                alert_id = f"vol_{current_week}_{hash(company) % 10000}"
                if alert_id in state.emitted_alerts:
                    continue

                alerts.append(Alert(
                    alert_id=alert_id,
                    alert_type='volume_spike',
                    severity=severity,
                    week=current_week,
                    description=f"{company}: {observed} complaints vs {expected:.1f} expected ({ratio:.1f}x)",
                    evidence={
                        'entity': company,
                        'observed': int(observed),
                        'expected': round(expected, 1),
                        'ratio': round(ratio, 2),
                        'p_value': round(p_value, 6)
                    },
                    score=(-np.log10(p_value + 1e-10)) * ratio
                ))
                state.emitted_alerts.add(alert_id)

        return alerts


class ClusterDetector:
    """
    Detects clusters growing or shrinking faster than expected.

    Method: Z-test on proportions
    - Compare current week's cluster share to baseline share
    - Flag if |z| > threshold

    Why proportion test: Cluster share is bounded 0-1, z-test is standard.
    """

    def __init__(self, z_threshold: float = 3.0):
        self.z_threshold = z_threshold

    def detect(self, current_week: str, state: DetectorState) -> List[Alert]:
        if not state.has_sufficient_history(current_week):
            return []

        alerts = []
        current_data = state.weekly_data[current_week]
        baseline_weeks = state.get_baseline_weeks(current_week)
        baseline_data = pd.concat([state.weekly_data[w] for w in baseline_weeks])

        n_current = len(current_data)
        n_baseline = len(baseline_data)

        if n_current < 10 or n_baseline < 10:
            return []

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

            if se == 0:
                continue

            z = (p_current - p_baseline) / se

            if abs(z) > self.z_threshold:
                direction = 'growing' if z > 0 else 'shrinking'
                alert_type = f'cluster_{direction}'
                severity = 'high' if abs(z) > 5 else 'medium'

                cluster_label = state.cluster_labels.get(cluster_id, f'cluster_{cluster_id}')
                alert_id = f"clust_{current_week}_{cluster_id}"

                if alert_id in state.emitted_alerts:
                    continue

                alerts.append(Alert(
                    alert_id=alert_id,
                    alert_type=alert_type,
                    severity=severity,
                    week=current_week,
                    description=f"Cluster '{cluster_label[:50]}' {direction}: {p_baseline*100:.1f}% → {p_current*100:.1f}%",
                    evidence={
                        'cluster_id': int(cluster_id),
                        'cluster_label': cluster_label,
                        'baseline_share': round(p_baseline * 100, 2),
                        'current_share': round(p_current * 100, 2),
                        'z_score': round(z, 2)
                    },
                    score=abs(z)
                ))
                state.emitted_alerts.add(alert_id)

        return alerts


class CorrelationDetector:
    """
    Detects when multiple companies spike together (systemic issue).

    Method: Count companies with >2x growth, flag if >= threshold

    Why this matters: Correlated spikes suggest external cause
    (regulation change, system outage, widespread scam)
    """

    def __init__(self, min_companies: int = 3, growth_threshold: float = 2.0):
        self.min_companies = min_companies
        self.growth_threshold = growth_threshold

    def detect(self, current_week: str, state: DetectorState) -> List[Alert]:
        if not state.has_sufficient_history(current_week):
            return []

        current_data = state.weekly_data[current_week]
        baseline_weeks = state.get_baseline_weeks(current_week)
        baseline_data = pd.concat([state.weekly_data[w] for w in baseline_weeks])

        current_counts = current_data.groupby('company').size()
        baseline_avg = baseline_data.groupby('company').size() / len(baseline_weeks)

        # Find companies with significant growth
        growth_ratios = {}
        for company in current_counts.index:
            if company in baseline_avg.index and baseline_avg[company] >= 2:
                ratio = current_counts[company] / baseline_avg[company]
                if ratio > self.growth_threshold:
                    growth_ratios[company] = ratio

        if len(growth_ratios) < self.min_companies:
            return []

        alert_id = f"corr_{current_week}"
        if alert_id in state.emitted_alerts:
            return []

        # Find common issues among spiking companies
        spiking_data = current_data[current_data['company'].isin(growth_ratios.keys())]
        common_issues = spiking_data['issue'].value_counts().head(3).to_dict() if 'issue' in spiking_data.columns else {}

        severity = 'high' if len(growth_ratios) >= 5 else 'medium'

        alert = Alert(
            alert_id=alert_id,
            alert_type='correlated_spike',
            severity=severity,
            week=current_week,
            description=f"{len(growth_ratios)} companies spiked together (avg {np.mean(list(growth_ratios.values())):.1f}x)",
            evidence={
                'companies_affected': len(growth_ratios),
                'company_list': list(growth_ratios.keys())[:10],
                'avg_growth_ratio': round(np.mean(list(growth_ratios.values())), 2),
                'common_issues': common_issues
            },
            score=len(growth_ratios) * np.mean(list(growth_ratios.values()))
        )
        state.emitted_alerts.add(alert_id)
        return [alert]


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class EmergingPatternDetector:
    """
    Main orchestrator that processes batches and emits alerts.

    Usage:
        detector = EmergingPatternDetector()
        detector.initialize(historical_df)  # Load historical data
        alerts = detector.process_batch(new_df)  # Process new batch

    State Management:
        - State is maintained between process_batch() calls
        - In production, state would be persisted to database
        - Current implementation keeps state in memory
    """

    def __init__(self):
        self.state = DetectorState()
        self.feature_extractor = FeatureExtractor()
        self.detectors = [
            VolumeDetector(),
            ClusterDetector(),
            CorrelationDetector()
        ]

    def initialize(self, df: pd.DataFrame) -> 'EmergingPatternDetector':
        """
        Initialize detector with historical data.

        This:
        1. Fits the text vectorizer and clustering model
        2. Computes cluster assignments for all historical data
        3. Organizes data by week for baseline computation
        """
        print("Initializing detector...")

        # Standardize column names
        df = self._standardize_columns(df)

        # Fit feature extractor on all data
        print("  Fitting text model...")
        df['cluster'] = self.feature_extractor.fit(df['text'], self.state)

        # Organize by week
        print("  Organizing by week...")
        df['week'] = df['timestamp'].dt.to_period('W').astype(str)
        for week in sorted(df['week'].unique()):
            self.state.weekly_data[week] = df[df['week'] == week].copy()

        print(f"  Initialized with {len(df)} complaints across {len(self.state.weekly_data)} weeks")
        print(f"  Clusters: {len(self.state.cluster_labels)}")

        return self

    def process_batch(self, df: pd.DataFrame) -> List[Alert]:
        """
        Process a new batch of complaints.

        In production:
        - This would be called when new data arrives (e.g., weekly)
        - Updates internal state
        - Returns alerts for the new batch
        """
        df = self._standardize_columns(df)

        # Transform using fitted models
        _, df['cluster'] = self.feature_extractor.transform(df['text'], self.state)

        # Add to weekly data
        df['week'] = df['timestamp'].dt.to_period('W').astype(str)
        for week in df['week'].unique():
            week_data = df[df['week'] == week]
            if week in self.state.weekly_data:
                self.state.weekly_data[week] = pd.concat([
                    self.state.weekly_data[week], week_data
                ])
            else:
                self.state.weekly_data[week] = week_data.copy()

        # Run detectors on new weeks
        all_alerts = []
        for week in df['week'].unique():
            for detector in self.detectors:
                alerts = detector.detect(week, self.state)
                all_alerts.extend(alerts)

        # Sort by score
        all_alerts.sort(key=lambda x: x.score, reverse=True)

        return all_alerts

    def run_detection_on_all_weeks(self) -> List[Alert]:
        """
        Run detection on all weeks in state.

        Used for batch analysis of historical data.
        """
        all_alerts = []
        weeks = sorted(self.state.weekly_data.keys())

        print(f"Running detection on {len(weeks)} weeks...")
        for week in weeks:
            for detector in self.detectors:
                alerts = detector.detect(week, self.state)
                all_alerts.extend(alerts)

        all_alerts.sort(key=lambda x: x.score, reverse=True)
        print(f"Generated {len(all_alerts)} alerts")

        return all_alerts

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map various column names to standard schema."""
        column_mapping = {
            'Date received': 'timestamp',
            'Consumer complaint narrative': 'text',
            'Company': 'company',
            'Product': 'product',
            'Sub-product': 'sub_product',
            'Issue': 'issue',
            'State': 'state',
            'Complaint ID': 'complaint_id',
            'Submitted via': 'channel'
        }

        df = df.copy()
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df[new_name] = df[old_name]

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        return df

    def get_alert_summary(self, alerts: List[Alert]) -> Dict:
        """Generate summary statistics for alerts."""
        if not alerts:
            return {'total': 0}

        df = pd.DataFrame([a.to_dict() for a in alerts])
        return {
            'total': len(alerts),
            'by_type': df['type'].value_counts().to_dict(),
            'by_severity': df['severity'].value_counts().to_dict(),
            'top_entities': df[df['type'] == 'volume_spike']['entity'].value_counts().head(5).to_dict() if 'entity' in df.columns else {}
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run detector on CFPB complaint data."""
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else 'complaints-2026-02-01_19_29.csv'

    print("=" * 60)
    print("EMERGING PATTERN DETECTOR")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"Loaded {len(df):,} complaints")

    # Initialize detector
    detector = EmergingPatternDetector()
    detector.initialize(df)

    # Run detection
    alerts = detector.run_detection_on_all_weeks()

    # Summary
    summary = detector.get_alert_summary(alerts)
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total alerts: {summary['total']}")
    print(f"\nBy type: {summary['by_type']}")
    print(f"\nBy severity: {summary['by_severity']}")

    # Save alerts
    if alerts:
        alerts_df = pd.DataFrame([a.to_dict() for a in alerts])
        alerts_df.to_csv('alerts_v2.csv', index=False)
        print("\nAlerts saved to alerts_v2.csv")

    # Print top alerts
    print("\n" + "=" * 60)
    print("TOP 10 ALERTS")
    print("=" * 60)
    for alert in alerts[:10]:
        print(f"\n[{alert.severity.upper()}] {alert.alert_type}")
        print(f"  Week: {alert.week}")
        print(f"  {alert.description}")

    return detector, alerts


if __name__ == '__main__':
    detector, alerts = main()
