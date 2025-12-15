import pandas as pd
df = pd.read_parquet('data/processed/mag7_news_5_topics_clean.parquet')
print(df.columns)
