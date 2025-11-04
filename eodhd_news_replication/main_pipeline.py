#!/usr/bin/env python3
"""
Main Pipeline Orchestrator
Runs the complete EODHD news event study pipeline
"""

import sys
import os
import logging
import argparse
from datetime import datetime
from pathlib import Path
import yaml
import json

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import pipeline modules
from src.data.data_acquisition import EODHDDataAcquisition
from src.data.text_preprocessing import TextPreprocessor
from src.data.document_term_matrix import build_dtm_pipeline
from src.models.lda_topic_modeling import run_lda_pipeline
from src.analysis.time_aggregation import run_time_aggregation_pipeline
from src.models.predictive_models import run_predictive_models_pipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class NewsEventStudyPipeline:
    """Main pipeline orchestrator for news event study"""

    def __init__(self, config_path: str = "conf/experiment.yaml"):
        """Initialize pipeline with configuration"""
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.pipeline_state = {
            'start_time': datetime.now().isoformat(),
            'steps_completed': [],
            'errors': []
        }

    def run_data_acquisition(self, skip_if_exists: bool = True):
        """Step 1: Data Acquisition"""
        logger.info("=" * 50)
        logger.info("STEP 1: DATA ACQUISITION")
        logger.info("=" * 50)

        # Check if data already exists
        if skip_if_exists:
            raw_files = list(Path("data/raw").glob("news_raw_*.parquet"))
            if raw_files:
                logger.info(f"Found {len(raw_files)} existing raw data files. Skipping acquisition.")
                self.pipeline_state['steps_completed'].append('data_acquisition_skipped')
                return

        try:
            acquisition = EODHDDataAcquisition(config_path=self.config_path)

            # Fetch news data
            logger.info("Fetching news data...")
            news_df = acquisition.fetch_all_news()

            # Fetch price data
            logger.info("Fetching price data...")
            prices_df = acquisition.fetch_all_prices()

            # Calculate returns
            if not prices_df.empty:
                acquisition.calculate_returns(prices_df)

            self.pipeline_state['steps_completed'].append('data_acquisition')
            logger.info("Data acquisition complete!")

        except Exception as e:
            logger.error(f"Error in data acquisition: {e}")
            self.pipeline_state['errors'].append(f"data_acquisition: {str(e)}")
            raise

    def run_text_preprocessing(self):
        """Step 2: Text Preprocessing"""
        logger.info("=" * 50)
        logger.info("STEP 2: TEXT PREPROCESSING")
        logger.info("=" * 50)

        try:
            preprocessor = TextPreprocessor(config_path=self.config_path)
            df_processed, vocabulary = preprocessor.process_pipeline()

            self.pipeline_state['steps_completed'].append('text_preprocessing')
            logger.info("Text preprocessing complete!")

        except Exception as e:
            logger.error(f"Error in text preprocessing: {e}")
            self.pipeline_state['errors'].append(f"text_preprocessing: {str(e)}")
            raise

    def run_dtm_construction(self):
        """Step 3: Document-Term Matrix Construction"""
        logger.info("=" * 50)
        logger.info("STEP 3: DOCUMENT-TERM MATRIX")
        logger.info("=" * 50)

        try:
            dtm, doc_index, term_index = build_dtm_pipeline()

            self.pipeline_state['steps_completed'].append('dtm_construction')
            logger.info("DTM construction complete!")

        except Exception as e:
            logger.error(f"Error in DTM construction: {e}")
            self.pipeline_state['errors'].append(f"dtm_construction: {str(e)}")
            raise

    def run_topic_modeling(self, n_topics: int = None):
        """Step 4: LDA Topic Modeling"""
        logger.info("=" * 50)
        logger.info("STEP 4: LDA TOPIC MODELING")
        logger.info("=" * 50)

        try:
            results = run_lda_pipeline(n_topics)

            self.pipeline_state['steps_completed'].append('topic_modeling')
            logger.info("Topic modeling complete!")

            return results

        except Exception as e:
            logger.error(f"Error in topic modeling: {e}")
            self.pipeline_state['errors'].append(f"topic_modeling: {str(e)}")
            raise

    def run_time_aggregation(self, model_name: str = None):
        """Step 5: Time Aggregation"""
        logger.info("=" * 50)
        logger.info("STEP 5: TIME AGGREGATION")
        logger.info("=" * 50)

        try:
            all_series = run_time_aggregation_pipeline(model_name)

            self.pipeline_state['steps_completed'].append('time_aggregation')
            logger.info("Time aggregation complete!")

            return all_series

        except Exception as e:
            logger.error(f"Error in time aggregation: {e}")
            self.pipeline_state['errors'].append(f"time_aggregation: {str(e)}")
            raise

    def run_predictive_models(self):
        """Step 6: Predictive Models and VAR"""
        logger.info("=" * 50)
        logger.info("STEP 6: PREDICTIVE MODELS")
        logger.info("=" * 50)

        try:
            results = run_predictive_models_pipeline()

            self.pipeline_state['steps_completed'].append('predictive_models')
            logger.info("Predictive modeling complete!")

            return results

        except Exception as e:
            logger.error(f"Error in predictive models: {e}")
            self.pipeline_state['errors'].append(f"predictive_models: {str(e)}")
            raise

    def generate_summary_report(self):
        """Generate summary report of pipeline results"""
        logger.info("=" * 50)
        logger.info("GENERATING SUMMARY REPORT")
        logger.info("=" * 50)

        report = {
            'pipeline_run': self.pipeline_state,
            'data_summary': {},
            'model_summary': {},
            'results_summary': {}
        }

        # Data summary
        try:
            # Check for processed data
            if Path("data/clean/news_clean.parquet").exists():
                import pandas as pd
                df = pd.read_parquet("data/clean/news_clean.parquet")
                report['data_summary']['n_documents'] = len(df)
                report['data_summary']['date_range'] = f"{df['published_at'].min()} to {df['published_at'].max()}"

            # Check for DTM
            if Path("logs/dtm_stats.json").exists():
                with open("logs/dtm_stats.json", 'r') as f:
                    report['data_summary']['dtm_stats'] = json.load(f)

        except Exception as e:
            logger.warning(f"Error loading data summary: {e}")

        # Model summary
        try:
            # LDA results
            model_dirs = list(Path("models").glob("lda_k=*/metrics.json"))
            if model_dirs:
                with open(model_dirs[0], 'r') as f:
                    report['model_summary']['lda'] = json.load(f)

            # VAR results
            if Path("results/var_summary.json").exists():
                with open("results/var_summary.json", 'r') as f:
                    report['model_summary']['var'] = json.load(f)

        except Exception as e:
            logger.warning(f"Error loading model summary: {e}")

        # Results summary
        try:
            if Path("results/trading_performance.json").exists():
                with open("results/trading_performance.json", 'r') as f:
                    report['results_summary']['trading'] = json.load(f)

        except Exception as e:
            logger.warning(f"Error loading results summary: {e}")

        # Save report
        report_file = f"results/pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Summary report saved to {report_file}")

        # Print summary
        print("\n" + "=" * 50)
        print("PIPELINE SUMMARY")
        print("=" * 50)
        print(f"Steps completed: {', '.join(report['pipeline_run']['steps_completed'])}")

        if report['pipeline_run']['errors']:
            print(f"Errors encountered: {len(report['pipeline_run']['errors'])}")

        if 'n_documents' in report['data_summary']:
            print(f"Documents processed: {report['data_summary']['n_documents']}")

        if 'lda' in report['model_summary']:
            print(f"Topics discovered: {report['model_summary']['lda']['n_topics']}")
            print(f"Topic coherence: {report['model_summary']['lda']['coherence']:.4f}")

        if 'trading' in report['results_summary']:
            trading = report['results_summary']['trading']
            print(f"Strategy Sharpe Ratio: {trading['strategy_sharpe']:.3f}")
            print(f"Hit Rate: {trading['hit_rate']:.3f}")
            print(f"Max Drawdown: {trading['max_drawdown']:.3f}")

        return report

    def run_full_pipeline(self, skip_data_acquisition: bool = False,
                         n_topics: int = None):
        """Run the complete pipeline"""
        logger.info("Starting EODHD News Event Study Pipeline")
        logger.info(f"Configuration: {self.config_path}")

        # Step 1: Data Acquisition
        if not skip_data_acquisition:
            self.run_data_acquisition()
        else:
            logger.info("Skipping data acquisition as requested")

        # Step 2: Text Preprocessing
        self.run_text_preprocessing()

        # Step 3: DTM Construction
        self.run_dtm_construction()

        # Step 4: Topic Modeling
        self.run_topic_modeling(n_topics)

        # Step 5: Time Aggregation
        self.run_time_aggregation()

        # Step 6: Predictive Models
        self.run_predictive_models()

        # Generate summary report
        report = self.generate_summary_report()

        logger.info("Pipeline complete!")

        return report


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='EODHD News Event Study Pipeline')
    parser.add_argument('--config', type=str, default='conf/experiment.yaml',
                       help='Path to configuration file')
    parser.add_argument('--skip-data-acquisition', action='store_true',
                       help='Skip data acquisition if raw data exists')
    parser.add_argument('--n-topics', type=int, default=None,
                       help='Number of topics for LDA (overrides config)')
    parser.add_argument('--steps', type=str, nargs='+',
                       choices=['data', 'preprocess', 'dtm', 'lda', 'aggregate', 'predict'],
                       help='Run only specific pipeline steps')

    args = parser.parse_args()

    # Create pipeline
    pipeline = NewsEventStudyPipeline(args.config)

    if args.steps:
        # Run specific steps
        logger.info(f"Running specific steps: {args.steps}")

        if 'data' in args.steps:
            pipeline.run_data_acquisition(skip_if_exists=args.skip_data_acquisition)

        if 'preprocess' in args.steps:
            pipeline.run_text_preprocessing()

        if 'dtm' in args.steps:
            pipeline.run_dtm_construction()

        if 'lda' in args.steps:
            pipeline.run_topic_modeling(args.n_topics)

        if 'aggregate' in args.steps:
            pipeline.run_time_aggregation()

        if 'predict' in args.steps:
            pipeline.run_predictive_models()

        # Generate report
        pipeline.generate_summary_report()

    else:
        # Run full pipeline
        pipeline.run_full_pipeline(
            skip_data_acquisition=args.skip_data_acquisition,
            n_topics=args.n_topics
        )


if __name__ == "__main__":
    main()