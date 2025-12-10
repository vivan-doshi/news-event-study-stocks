
import argparse
import subprocess
import os
import sys

# Default paths
# data/raw exists? No, inputs are in data/processed from previous steps?
# The inputs to feature engineering are:
# 1. NEW_DATA: data/processed/mag7_news_with_sentiment_and_topics_labeledV2.parquet
# 2. STOCK_DATA: data/processed/mag7_yf_2021_2025.parquet
# 3. MAP: config/topic_to_label_map_v2.csv

# Output of FE:
# data/processed/mag7_aggregated_features.parquet

# Output of Event Study:
# data/outputs/results/

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

DEFAULT_NEWS_PATH = os.path.join(DATA_DIR, 'processed', 'mag7_news_with_sentiment_and_topics_labeledV2.parquet')
DEFAULT_STOCK_PATH = os.path.join(DATA_DIR, 'processed', 'mag7_yf_2021_2025.parquet')
DEFAULT_MAP_PATH = os.path.join(CONFIG_DIR, 'topic_to_label_map_v2.csv')
DEFAULT_AGG_PATH = os.path.join(DATA_DIR, 'processed', 'mag7_aggregated_features.parquet')
DEFAULT_RESULTS_DIR = os.path.join(DATA_DIR, 'outputs', 'results')

def run_feature_engineering(news_path, stock_path, map_path, output_path):
    print("\n=== Running Feature Engineering Stage ===")
    script_path = os.path.join(SRC_DIR, 'analysis', 'feature_engineering.py')
    cmd = [
        sys.executable, script_path,
        '--news_path', news_path,
        '--stock_path', stock_path,
        '--map_path', map_path,
        '--output_path', output_path
    ]
    print(f"Executing: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def run_event_study(data_path, output_dir, symbols=None, target=None):
    print("\n=== Running Event Study Stage ===")
    script_path = os.path.join(SRC_DIR, 'analysis', 'event_study.py')
    cmd = [
        sys.executable, script_path,
        '--data_path', data_path,
        '--output_dir', output_dir
    ]
    if symbols:
        cmd.extend(['--symbols', symbols])
    if target:
        cmd.extend(['--target', target])
        
    print(f"Executing: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def main():
    parser = argparse.ArgumentParser(description="Mag7 News Event Study Pipeline")
    parser.add_argument('--stage', choices=['all', 'feature_engineering', 'event_study'], default='all', help='Pipeline stage to run')
    parser.add_argument('--symbol', help='Specific symbol(s) for event study (comma separated)')
    
    # input overrides
    parser.add_argument('--news_path', default=DEFAULT_NEWS_PATH)
    parser.add_argument('--stock_path', default=DEFAULT_STOCK_PATH)
    parser.add_argument('--map_path', default=DEFAULT_MAP_PATH)
    parser.add_argument('--agg_path', default=DEFAULT_AGG_PATH)
    parser.add_argument('--output_dir', default=DEFAULT_RESULTS_DIR)
    
    args = parser.parse_args()
    
    if args.stage in ['all', 'feature_engineering']:
        run_feature_engineering(args.news_path, args.stock_path, args.map_path, args.agg_path)
        
    if args.stage in ['all', 'event_study']:
        # If running all, ensure agg_path exists (it should be created by FE)
        if args.stage == 'event_study' and not os.path.exists(args.agg_path):
             print(f"Error: Aggregated data not found at {args.agg_path}. Run feature_engineering first.")
             sys.exit(1)
             
        run_event_study(args.agg_path, args.output_dir, args.symbol)

if __name__ == "__main__":
    main()
