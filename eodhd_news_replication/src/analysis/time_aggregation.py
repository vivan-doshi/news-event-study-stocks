"""
Time Aggregation for Topic Attention Series
Aggregates document-level topics to time series
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class TopicTimeAggregation:
    """Aggregate topic weights into time series"""

    def __init__(self, config_path: str = "conf/experiment.yaml"):
        """Initialize time aggregation with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.frequencies = self.config['time_aggregation']['frequencies']
        self.methods = self.config['time_aggregation']['aggregation_methods']
        self.smoothing_window = self.config['time_aggregation']['smoothing_window']

    def load_topic_data(self, model_name: str = None, online: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load document-topic distributions and document index"""
        # Find model directory if not specified
        if model_name is None:
            model_dirs = list(Path("models").glob("lda_k=*"))
            if not model_dirs:
                raise FileNotFoundError("No LDA models found")
            # Use model with highest K by default
            model_name = sorted(model_dirs)[-1].name

        # Load document-topics
        if online:
            theta_file = Path(f"models/{model_name}_online/theta_online.parquet")
        else:
            theta_file = Path(f"models/{model_name}/theta.parquet")

        if not theta_file.exists():
            raise FileNotFoundError(f"Topic file not found: {theta_file}")

        theta_df = pd.read_parquet(theta_file)

        # Load document index
        doc_index = pd.read_parquet("data/derived/doc_index.parquet")

        # Load clean data for document lengths
        clean_df = pd.read_parquet("data/clean/news_clean.parquet")
        clean_df['doc_length'] = clean_df['tokens'].apply(len)

        # Merge
        theta_df = theta_df.merge(doc_index[['article_id', 'published_at']], on='article_id')
        theta_df = theta_df.merge(clean_df[['article_id', 'doc_length']], on='article_id', how='left')

        # Parse dates
        theta_df['published_date'] = pd.to_datetime(theta_df['published_at'])

        logger.info(f"Loaded topic data for {len(theta_df)} documents from {model_name}")

        return theta_df, doc_index

    def aggregate_topics(self, theta_df: pd.DataFrame, frequency: str = 'daily',
                        method: str = 'mean') -> pd.DataFrame:
        """Aggregate topics to specified time frequency"""
        logger.info(f"Aggregating topics to {frequency} using {method} method")

        # Get topic columns
        topic_cols = [col for col in theta_df.columns if col.startswith('topic_')]

        # Set date as index
        theta_df = theta_df.set_index('published_date')

        if method == 'mean':
            # Simple mean aggregation
            if frequency == 'daily':
                aggregated = theta_df.groupby(pd.Grouper(freq='D'))[topic_cols].mean()
            elif frequency == 'monthly':
                aggregated = theta_df.groupby(pd.Grouper(freq='M'))[topic_cols].mean()
            else:
                raise ValueError(f"Unknown frequency: {frequency}")

        elif method == 'length_weighted':
            # Document length weighted average
            if 'doc_length' not in theta_df.columns:
                logger.warning("Document lengths not available, using simple mean")
                return self.aggregate_topics(theta_df.reset_index(), frequency, 'mean')

            # Weight by document length
            for topic in topic_cols:
                theta_df[f"{topic}_weighted"] = theta_df[topic] * theta_df['doc_length']

            weighted_cols = [f"{col}_weighted" for col in topic_cols]

            if frequency == 'daily':
                weighted_sums = theta_df.groupby(pd.Grouper(freq='D'))[weighted_cols].sum()
                length_sums = theta_df.groupby(pd.Grouper(freq='D'))['doc_length'].sum()
            elif frequency == 'monthly':
                weighted_sums = theta_df.groupby(pd.Grouper(freq='M'))[weighted_cols].sum()
                length_sums = theta_df.groupby(pd.Grouper(freq='M'))['doc_length'].sum()
            else:
                raise ValueError(f"Unknown frequency: {frequency}")

            # Calculate weighted average
            aggregated = pd.DataFrame(index=weighted_sums.index)
            for topic in topic_cols:
                aggregated[topic] = weighted_sums[f"{topic}_weighted"] / length_sums

        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        # Fill missing dates with zeros
        aggregated = aggregated.fillna(0)

        # Add metadata
        aggregated['n_documents'] = theta_df.groupby(pd.Grouper(freq='D' if frequency == 'daily' else 'M')).size()

        logger.info(f"Created {frequency} series with shape {aggregated.shape}")

        return aggregated

    def add_smoothing(self, series: pd.DataFrame, window: int = None) -> pd.DataFrame:
        """Add smoothed version of series for visualization"""
        if window is None:
            window = self.smoothing_window

        # Get topic columns
        topic_cols = [col for col in series.columns if col.startswith('topic_')]

        # Create smoothed versions
        smoothed = series.copy()
        for col in topic_cols:
            smoothed[f"{col}_smoothed"] = series[col].rolling(window=window, center=True).mean()

        return smoothed

    def calculate_statistics(self, series: pd.DataFrame) -> pd.DataFrame:
        """Calculate statistics for topic attention series"""
        topic_cols = [col for col in series.columns if col.startswith('topic_') and not col.endswith('_smoothed')]

        stats = pd.DataFrame({
            'mean': series[topic_cols].mean(),
            'std': series[topic_cols].std(),
            'min': series[topic_cols].min(),
            'max': series[topic_cols].max(),
            'cv': series[topic_cols].std() / series[topic_cols].mean(),  # Coefficient of variation
            'autocorr_1': [series[col].autocorr(1) for col in topic_cols],  # Lag-1 autocorrelation
            'n_spikes': [(series[col] > series[col].mean() + 2 * series[col].std()).sum() for col in topic_cols]
        })

        stats.index.name = 'topic'
        return stats

    def merge_with_returns(self, topic_series: pd.DataFrame, frequency: str = 'monthly') -> pd.DataFrame:
        """Merge topic series with market returns"""
        # Load returns data
        if frequency == 'daily':
            returns_file = "data/derived/returns_daily.parquet"
        else:
            returns_file = "data/derived/returns_monthly.parquet"

        if Path(returns_file).exists():
            returns_df = pd.read_parquet(returns_file)

            # Filter for SPY
            spy_returns = returns_df[returns_df['symbol'] == 'SPY.US'][['log_return', 'simple_return']]

            if not spy_returns.empty:
                # Merge
                panel = topic_series.merge(spy_returns, left_index=True, right_index=True, how='inner')

                logger.info(f"Created panel with {len(panel)} observations")

                return panel
            else:
                logger.warning("No SPY returns found")
                return topic_series
        else:
            logger.warning(f"Returns file not found: {returns_file}")
            return topic_series

    def create_all_series(self, model_name: str = None) -> Dict[str, pd.DataFrame]:
        """Create all topic attention series variations"""
        results = {}

        # Load data for both batch and online models
        for online in [False, True]:
            model_type = "online" if online else "batch"

            try:
                theta_df, doc_index = self.load_topic_data(model_name, online=online)

                for frequency in self.frequencies:
                    for method in self.methods:
                        # Aggregate topics
                        series = self.aggregate_topics(theta_df, frequency, method)

                        # Add smoothing for monthly series
                        if frequency == 'monthly':
                            series = self.add_smoothing(series)

                        # Calculate statistics
                        stats = self.calculate_statistics(series)

                        # Save series
                        output_file = f"data/derived/topics_{frequency}_{method}_{model_type}.parquet"
                        series.to_parquet(output_file, compression='snappy')
                        logger.info(f"Saved {output_file}")

                        # Save statistics
                        stats_file = f"data/derived/topics_{frequency}_{method}_{model_type}_stats.csv"
                        stats.to_csv(stats_file)

                        # Merge with returns and create panel
                        panel = self.merge_with_returns(series, frequency)
                        panel_file = f"data/derived/panel_{frequency}_{method}_{model_type}.parquet"
                        panel.to_parquet(panel_file, compression='snappy')

                        results[f"{frequency}_{method}_{model_type}"] = series

            except FileNotFoundError as e:
                logger.warning(f"Could not process {model_type} model: {e}")

        return results

    def identify_event_spikes(self, series: pd.DataFrame, threshold_std: float = 2.0) -> pd.DataFrame:
        """Identify dates with significant topic spikes"""
        topic_cols = [col for col in series.columns if col.startswith('topic_') and not col.endswith('_smoothed')]

        spikes = []

        for col in topic_cols:
            # Calculate z-scores
            z_scores = (series[col] - series[col].mean()) / series[col].std()

            # Find spikes
            spike_dates = z_scores[z_scores > threshold_std].index

            for date in spike_dates:
                spikes.append({
                    'date': date,
                    'topic': col,
                    'attention': series.loc[date, col],
                    'z_score': z_scores.loc[date]
                })

        spikes_df = pd.DataFrame(spikes)

        if not spikes_df.empty:
            spikes_df = spikes_df.sort_values(['date', 'z_score'], ascending=[True, False])

        return spikes_df

    def export_for_visualization(self, series: pd.DataFrame, output_file: str):
        """Export series in format suitable for visualization"""
        # Reset index to have date as column
        export_df = series.reset_index()
        export_df.rename(columns={'index': 'date'}, inplace=True)

        # Convert to long format for easier plotting
        topic_cols = [col for col in series.columns if col.startswith('topic_') and not col.endswith('_smoothed')]

        long_df = export_df.melt(
            id_vars=['date'],
            value_vars=topic_cols,
            var_name='topic',
            value_name='attention'
        )

        # Save
        long_df.to_csv(output_file, index=False)
        logger.info(f"Exported visualization data to {output_file}")


def run_time_aggregation_pipeline(model_name: str = None) -> Dict:
    """Complete time aggregation pipeline"""
    logger.info("Starting time aggregation pipeline")

    # Initialize aggregator
    aggregator = TopicTimeAggregation()

    # Create all series
    all_series = aggregator.create_all_series(model_name)

    # Identify event spikes in monthly series
    if 'monthly_mean_batch' in all_series:
        monthly_series = all_series['monthly_mean_batch']
        spikes = aggregator.identify_event_spikes(monthly_series)

        if not spikes.empty:
            spikes.to_csv("results/topic_spikes.csv", index=False)
            logger.info(f"Identified {len(spikes)} topic spikes")

            # Show top spikes
            logger.info("\nTop 10 topic spikes:")
            print(spikes.head(10))

        # Export for visualization
        aggregator.export_for_visualization(
            monthly_series,
            "results/topic_attention_monthly.csv"
        )

    logger.info("Time aggregation pipeline complete!")

    return all_series


def main():
    """Main function to run time aggregation"""
    logging.basicConfig(level=logging.INFO)
    all_series = run_time_aggregation_pipeline()

    # Show sample of monthly series
    if 'monthly_mean_batch' in all_series:
        monthly = all_series['monthly_mean_batch']
        logger.info(f"\nMonthly topic attention shape: {monthly.shape}")
        logger.info(f"Date range: {monthly.index.min()} to {monthly.index.max()}")
        logger.info(f"\nFirst few rows:\n{monthly.head()}")


if __name__ == "__main__":
    main()