"""
Emerging Pattern Detector - Incremental Multi-Product Version
==============================================================

A stateful, incremental detector that:
1. Processes batches of complaints as they arrive
2. Maintains internal state between batches (no reprocessing)
3. Handles multiple products
4. Uses sentence embeddings for semantic clustering

Usage:
    detector = Detector.load_or_create('state.pkl')
    alerts = detector.ingest_batch(new_complaints_df)
    detector.save('state.pkl')
"""

import pandas as pd
import numpy as np
import pickle
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from scipy import stats as scipy_stats
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# Sentence embeddings
from sentence_transformers import SentenceTransformer

# Clustering
from sklearn.cluster import MiniBatchKMeans


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Alert:
    """Alert emitted by the detector."""
    id: str
    type: str
    severity: str
    timestamp: datetime
    week: str
    product: Optional[str]
    entity: Optional[str]
    description: str
    evidence: Dict[str, Any]
    score: float
    complaint_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'type': self.type,
            'severity': self.severity,
            'timestamp': str(self.timestamp),
            'week': self.week,
            'product': self.product,
            'entity': self.entity,
            'description': self.description,
            'score': self.score,
            'complaint_ids': self.complaint_ids[:10],  # Sample
            **self.evidence
        }


@dataclass
class WeekStats:
    """Statistics for a single week, used for baseline computation."""
    week: str
    product: str
    total_count: int
    entity_counts: Dict[str, int]  # company -> count
    cluster_counts: Dict[int, int]  # cluster_id -> count
    embeddings_centroid: Optional[np.ndarray] = None
    complaint_ids: List[str] = field(default_factory=list)


class DetectorState:
    """
    Persistent state maintained between batch processing runs.

    This is what gets serialized to disk.
    """
    def __init__(self):
        # Embedding model (loaded separately, not serialized)
        self._embedder: Optional[SentenceTransformer] = None

        # Clustering model per product (serialized)
        self.cluster_models: Dict[str, MiniBatchKMeans] = {}
        self.cluster_labels: Dict[str, Dict[int, str]] = {}  # product -> {cluster_id -> label}

        # Historical stats: product -> week -> WeekStats
        self.weekly_stats: Dict[str, Dict[str, WeekStats]] = defaultdict(dict)

        # Seen complaints (for deduplication)
        self.seen_complaint_ids: set = set()

        # Emitted alerts (to avoid duplicates)
        self.emitted_alert_ids: set = set()

        # Configuration
        self.config = {
            'baseline_weeks': 4,
            'min_entity_baseline': 5,
            'volume_p_threshold': 0.01,
            'cluster_z_threshold': 3.0,
            'novelty_percentile': 95,
            'n_clusters': 25,
            'embedding_model': 'all-MiniLM-L6-v2'
        }

        # Track products we've seen
        self.known_products: set = set()

    @property
    def embedder(self) -> SentenceTransformer:
        """Lazy load embedding model."""
        if self._embedder is None:
            print(f"Loading embedding model: {self.config['embedding_model']}...")
            self._embedder = SentenceTransformer(self.config['embedding_model'])
        return self._embedder

    def get_baseline_weeks(self, product: str, current_week: str) -> List[str]:
        """Get the N weeks before current_week for baseline."""
        if product not in self.weekly_stats:
            return []
        all_weeks = sorted(self.weekly_stats[product].keys())
        if current_week not in all_weeks:
            # Find where it would be inserted
            idx = len([w for w in all_weeks if w < current_week])
        else:
            idx = all_weeks.index(current_week)
        start = max(0, idx - self.config['baseline_weeks'])
        return all_weeks[start:idx]

    def has_baseline(self, product: str, current_week: str) -> bool:
        """Check if we have enough history."""
        return len(self.get_baseline_weeks(product, current_week)) >= self.config['baseline_weeks']


# =============================================================================
# DETECTOR
# =============================================================================

class Detector:
    """
    Main detector class. Processes batches incrementally.

    Key methods:
    - ingest_batch(df): Process new complaints, emit alerts
    - save(path): Persist state to disk
    - load_or_create(path): Load existing state or create new
    """

    def __init__(self, state: Optional[DetectorState] = None):
        self.state = state or DetectorState()

    @classmethod
    def load_or_create(cls, path: str) -> 'Detector':
        """Load state from disk or create new detector."""
        path = Path(path)
        if path.exists():
            print(f"Loading state from {path}...")
            with open(path, 'rb') as f:
                state = pickle.load(f)
            return cls(state)
        else:
            print("Creating new detector...")
            return cls()

    def save(self, path: str):
        """Save state to disk."""
        # Don't serialize the embedding model
        embedder = self.state._embedder
        self.state._embedder = None

        with open(path, 'wb') as f:
            pickle.dump(self.state, f)

        self.state._embedder = embedder
        print(f"State saved to {path}")

    def ingest_batch(self, df: pd.DataFrame) -> List[Alert]:
        """
        Process a batch of new complaints.

        This is the main entry point. Call this when new data arrives.

        Args:
            df: DataFrame with columns: timestamp, text, product, company (optional), complaint_id

        Returns:
            List of alerts for this batch
        """
        # Standardize columns
        df = self._standardize(df)

        # Filter already-seen complaints
        new_mask = ~df['complaint_id'].isin(self.state.seen_complaint_ids)
        df = df[new_mask].copy()

        if len(df) == 0:
            print("No new complaints in batch")
            return []

        print(f"Processing {len(df)} new complaints...")

        # Add to seen set
        self.state.seen_complaint_ids.update(df['complaint_id'].tolist())

        # Parse timestamps and weeks
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['week'] = df['timestamp'].dt.to_period('W').astype(str)

        # Get embeddings for all texts
        print("  Computing embeddings...")
        embeddings = self.state.embedder.encode(
            df['text'].fillna('').tolist(),
            show_progress_bar=True,
            batch_size=64
        )
        df['embedding_idx'] = range(len(df))

        # Process by product
        all_alerts = []
        for product in df['product'].unique():
            product_df = df[df['product'] == product].copy()
            product_embeddings = embeddings[product_df['embedding_idx'].values]

            # Update/create cluster model for this product
            self._update_clusters(product, product_df, product_embeddings)

            # Assign clusters
            product_df['cluster'] = self.state.cluster_models[product].predict(product_embeddings)

            # Update weekly stats
            self._update_stats(product, product_df, product_embeddings)

            # Run detection
            alerts = self._detect(product, product_df, product_embeddings)
            all_alerts.extend(alerts)

        # Sort by score
        all_alerts.sort(key=lambda a: a.score, reverse=True)

        print(f"  Generated {len(all_alerts)} alerts")
        return all_alerts

    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names."""
        df = df.copy()

        column_map = {
            'Date received': 'timestamp',
            'Consumer complaint narrative': 'text',
            'Product': 'product',
            'Company': 'company',
            'Complaint ID': 'complaint_id',
            'Sub-product': 'sub_product',
            'Issue': 'issue',
            'State': 'state',
            'Submitted via': 'channel'
        }

        for old, new in column_map.items():
            if old in df.columns and new not in df.columns:
                df[new] = df[old]

        # Ensure required columns
        if 'complaint_id' not in df.columns:
            df['complaint_id'] = [f"gen_{i}_{hash(str(row))}" for i, row in df.iterrows()]

        if 'company' not in df.columns:
            df['company'] = 'unknown'

        return df

    def _update_clusters(self, product: str, df: pd.DataFrame, embeddings: np.ndarray):
        """Update or create clustering model for a product."""
        n_clusters = self.state.config['n_clusters']

        if product not in self.state.cluster_models:
            # New product - create model
            print(f"  Creating cluster model for product: {product}")
            model = MiniBatchKMeans(
                n_clusters=min(n_clusters, len(df)),
                random_state=42,
                batch_size=256,
                n_init=3
            )
            model.fit(embeddings)
            self.state.cluster_models[product] = model
            self.state.known_products.add(product)

            # Generate cluster labels (we'll refine these over time)
            self.state.cluster_labels[product] = {
                i: f"cluster_{i}" for i in range(n_clusters)
            }
        else:
            # Existing product - partial fit (incremental update)
            self.state.cluster_models[product].partial_fit(embeddings)

    def _update_stats(self, product: str, df: pd.DataFrame, embeddings: np.ndarray):
        """Update weekly statistics for a product."""
        for week in df['week'].unique():
            week_df = df[df['week'] == week]
            week_embeddings = embeddings[week_df['embedding_idx'].values - df['embedding_idx'].min()]

            if week in self.state.weekly_stats[product]:
                # Update existing stats
                stats = self.state.weekly_stats[product][week]
                stats.total_count += len(week_df)

                # Update entity counts
                for company, count in week_df['company'].value_counts().items():
                    stats.entity_counts[company] = stats.entity_counts.get(company, 0) + count

                # Update cluster counts
                for cluster, count in week_df['cluster'].value_counts().items():
                    stats.cluster_counts[cluster] = stats.cluster_counts.get(cluster, 0) + count

                # Update centroid (weighted average)
                old_weight = stats.total_count - len(week_df)
                new_centroid = (
                    stats.embeddings_centroid * old_weight +
                    week_embeddings.mean(axis=0) * len(week_df)
                ) / stats.total_count
                stats.embeddings_centroid = new_centroid

                stats.complaint_ids.extend(week_df['complaint_id'].tolist())
            else:
                # Create new stats
                self.state.weekly_stats[product][week] = WeekStats(
                    week=week,
                    product=product,
                    total_count=len(week_df),
                    entity_counts=week_df['company'].value_counts().to_dict(),
                    cluster_counts=week_df['cluster'].value_counts().to_dict(),
                    embeddings_centroid=week_embeddings.mean(axis=0),
                    complaint_ids=week_df['complaint_id'].tolist()
                )

    def _detect(self, product: str, df: pd.DataFrame, embeddings: np.ndarray) -> List[Alert]:
        """Run all detectors for a product."""
        alerts = []

        for week in df['week'].unique():
            week_df = df[df['week'] == week]
            week_embeddings = embeddings[week_df['embedding_idx'].values - df['embedding_idx'].min()]

            if not self.state.has_baseline(product, week):
                continue

            # Volume detection
            alerts.extend(self._detect_volume(product, week, week_df))

            # Cluster detection
            alerts.extend(self._detect_clusters(product, week, week_df))

            # Novelty detection
            alerts.extend(self._detect_novelty(product, week, week_df, week_embeddings))

            # Cross-entity correlation
            alerts.extend(self._detect_correlation(product, week, week_df))

        return alerts

    def _detect_volume(self, product: str, week: str, df: pd.DataFrame) -> List[Alert]:
        """Detect volume spikes for entities."""
        alerts = []
        baseline_weeks = self.state.get_baseline_weeks(product, week)

        # Current counts
        current_counts = df['company'].value_counts()

        # Baseline average
        baseline_counts = defaultdict(list)
        for bw in baseline_weeks:
            stats = self.state.weekly_stats[product][bw]
            for company, count in stats.entity_counts.items():
                baseline_counts[company].append(count)

        baseline_avg = {c: np.mean(counts) for c, counts in baseline_counts.items()}

        # Test each company
        for company in current_counts.index:
            observed = current_counts[company]
            expected = baseline_avg.get(company, 0)

            if expected < self.state.config['min_entity_baseline'] / len(baseline_weeks):
                continue

            # Poisson test
            p_value = 1 - scipy_stats.poisson.cdf(observed - 1, expected)

            if p_value < self.state.config['volume_p_threshold']:
                ratio = observed / expected if expected > 0 else float('inf')
                severity = 'high' if ratio > 3 else 'medium' if ratio > 2 else 'low'

                alert_id = f"vol_{product}_{week}_{hashlib.md5(company.encode()).hexdigest()[:8]}"
                if alert_id in self.state.emitted_alert_ids:
                    continue

                alerts.append(Alert(
                    id=alert_id,
                    type='volume_spike',
                    severity=severity,
                    timestamp=datetime.now(),
                    week=week,
                    product=product,
                    entity=company,
                    description=f"{company}: {observed} vs {expected:.1f} expected ({ratio:.1f}x)",
                    evidence={'observed': int(observed), 'expected': round(expected, 1),
                              'ratio': round(ratio, 2), 'p_value': round(p_value, 6)},
                    score=(-np.log10(p_value + 1e-10)) * ratio,
                    complaint_ids=df[df['company'] == company]['complaint_id'].tolist()
                ))
                self.state.emitted_alert_ids.add(alert_id)

        return alerts

    def _detect_clusters(self, product: str, week: str, df: pd.DataFrame) -> List[Alert]:
        """Detect cluster growth/shrinkage."""
        alerts = []
        baseline_weeks = self.state.get_baseline_weeks(product, week)

        n_current = len(df)
        current_dist = df['cluster'].value_counts(normalize=True)

        # Baseline distribution
        baseline_counts = defaultdict(int)
        baseline_total = 0
        for bw in baseline_weeks:
            stats = self.state.weekly_stats[product][bw]
            baseline_total += stats.total_count
            for cluster, count in stats.cluster_counts.items():
                baseline_counts[cluster] += count

        if baseline_total < 10 or n_current < 10:
            return alerts

        baseline_dist = {c: count / baseline_total for c, count in baseline_counts.items()}

        for cluster_id in current_dist.index:
            p_current = current_dist[cluster_id]
            p_baseline = baseline_dist.get(cluster_id, 0)

            if p_baseline < 0.01:
                continue

            # Z-test
            p_pooled = (p_current * n_current + p_baseline * baseline_total) / (n_current + baseline_total)
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_current + 1/baseline_total))

            if se == 0:
                continue

            z = (p_current - p_baseline) / se

            if abs(z) > self.state.config['cluster_z_threshold']:
                direction = 'growing' if z > 0 else 'shrinking'
                severity = 'high' if abs(z) > 5 else 'medium'
                cluster_label = self.state.cluster_labels[product].get(cluster_id, f"cluster_{cluster_id}")

                alert_id = f"clust_{product}_{week}_{cluster_id}"
                if alert_id in self.state.emitted_alert_ids:
                    continue

                alerts.append(Alert(
                    id=alert_id,
                    type=f'cluster_{direction}',
                    severity=severity,
                    timestamp=datetime.now(),
                    week=week,
                    product=product,
                    entity=None,
                    description=f"Cluster '{cluster_label}' {direction}: {p_baseline*100:.1f}% → {p_current*100:.1f}%",
                    evidence={'cluster_id': int(cluster_id), 'cluster_label': cluster_label,
                              'baseline_share': round(p_baseline * 100, 2),
                              'current_share': round(p_current * 100, 2),
                              'z_score': round(z, 2)},
                    score=abs(z),
                    complaint_ids=df[df['cluster'] == cluster_id]['complaint_id'].tolist()
                ))
                self.state.emitted_alert_ids.add(alert_id)

        return alerts

    def _detect_novelty(self, product: str, week: str, df: pd.DataFrame,
                        embeddings: np.ndarray) -> List[Alert]:
        """Detect novel complaints that don't fit historical patterns."""
        alerts = []
        baseline_weeks = self.state.get_baseline_weeks(product, week)

        if len(baseline_weeks) == 0 or len(embeddings) < 10:
            return alerts

        # Get baseline centroid
        baseline_centroids = []
        for bw in baseline_weeks:
            stats = self.state.weekly_stats[product][bw]
            if stats.embeddings_centroid is not None:
                baseline_centroids.append(stats.embeddings_centroid)

        if not baseline_centroids:
            return alerts

        historical_centroid = np.mean(baseline_centroids, axis=0)

        # Compute distances
        distances = np.linalg.norm(embeddings - historical_centroid, axis=1)
        threshold = np.percentile(distances, self.state.config['novelty_percentile'])

        novel_mask = distances > threshold
        novel_count = novel_mask.sum()

        if novel_count > 10:
            alert_id = f"novel_{product}_{week}"
            if alert_id not in self.state.emitted_alert_ids:
                novel_ids = df.iloc[novel_mask]['complaint_id'].tolist()

                alerts.append(Alert(
                    id=alert_id,
                    type='novel_pattern',
                    severity='medium',
                    timestamp=datetime.now(),
                    week=week,
                    product=product,
                    entity=None,
                    description=f"{novel_count} novel complaints ({novel_count/len(df)*100:.1f}% of week)",
                    evidence={'novel_count': int(novel_count), 'total_count': len(df),
                              'novel_share': round(novel_count / len(df) * 100, 2)},
                    score=novel_count / len(df) * 10,
                    complaint_ids=novel_ids[:20]
                ))
                self.state.emitted_alert_ids.add(alert_id)

        return alerts

    def _detect_correlation(self, product: str, week: str, df: pd.DataFrame) -> List[Alert]:
        """Detect when multiple companies spike together."""
        alerts = []
        baseline_weeks = self.state.get_baseline_weeks(product, week)

        current_counts = df['company'].value_counts()

        # Baseline average
        baseline_counts = defaultdict(list)
        for bw in baseline_weeks:
            stats = self.state.weekly_stats[product][bw]
            for company, count in stats.entity_counts.items():
                baseline_counts[company].append(count)

        baseline_avg = {c: np.mean(counts) for c, counts in baseline_counts.items()}

        # Find spiking companies
        spiking = {}
        for company in current_counts.index:
            if company in baseline_avg and baseline_avg[company] >= 2:
                ratio = current_counts[company] / baseline_avg[company]
                if ratio > 2:
                    spiking[company] = ratio

        if len(spiking) >= 3:
            alert_id = f"corr_{product}_{week}"
            if alert_id not in self.state.emitted_alert_ids:
                spiking_df = df[df['company'].isin(spiking.keys())]
                common_issues = spiking_df['issue'].value_counts().head(3).to_dict() if 'issue' in df.columns else {}

                severity = 'high' if len(spiking) >= 5 else 'medium'

                alerts.append(Alert(
                    id=alert_id,
                    type='correlated_spike',
                    severity=severity,
                    timestamp=datetime.now(),
                    week=week,
                    product=product,
                    entity=None,
                    description=f"{len(spiking)} companies spiked together (avg {np.mean(list(spiking.values())):.1f}x)",
                    evidence={'companies_affected': len(spiking),
                              'company_list': list(spiking.keys())[:10],
                              'avg_ratio': round(np.mean(list(spiking.values())), 2),
                              'common_issues': common_issues},
                    score=len(spiking) * np.mean(list(spiking.values())),
                    complaint_ids=spiking_df['complaint_id'].tolist()[:50]
                ))
                self.state.emitted_alert_ids.add(alert_id)

        return alerts

    def get_summary(self) -> Dict:
        """Get summary of detector state."""
        return {
            'products': list(self.state.known_products),
            'total_complaints_seen': len(self.state.seen_complaint_ids),
            'total_alerts_emitted': len(self.state.emitted_alert_ids),
            'weeks_by_product': {
                p: len(weeks) for p, weeks in self.state.weekly_stats.items()
            }
        }


# =============================================================================
# CLI
# =============================================================================

def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python detector.py <data.csv> [state.pkl]")
        print("  data.csv: Input complaints (can be called multiple times for batches)")
        print("  state.pkl: State file (created if doesn't exist)")
        sys.exit(1)

    data_path = sys.argv[1]
    state_path = sys.argv[2] if len(sys.argv) > 2 else 'detector_state.pkl'

    print("=" * 60)
    print("EMERGING PATTERN DETECTOR")
    print("=" * 60)

    # Load or create detector
    detector = Detector.load_or_create(state_path)

    # Load data
    print(f"\nLoading {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"Loaded {len(df):,} rows")

    # Process
    alerts = detector.ingest_batch(df)

    # Save state
    detector.save(state_path)

    # Summary
    summary = detector.get_summary()
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Products: {summary['products']}")
    print(f"Total complaints processed: {summary['total_complaints_seen']:,}")
    print(f"Total alerts: {summary['total_alerts_emitted']}")
    print(f"Alerts this batch: {len(alerts)}")

    # Save alerts
    if alerts:
        alerts_df = pd.DataFrame([a.to_dict() for a in alerts])
        alerts_df.to_csv('alerts.csv', index=False)
        print(f"\nAlerts saved to alerts.csv")

        print("\n" + "=" * 60)
        print("TOP 10 ALERTS")
        print("=" * 60)
        for alert in alerts[:10]:
            print(f"\n[{alert.severity.upper()}] {alert.type} | {alert.product}")
            print(f"  Week: {alert.week}")
            print(f"  {alert.description}")

    return detector, alerts


if __name__ == '__main__':
    detector, alerts = main()
