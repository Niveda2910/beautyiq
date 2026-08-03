import pandas as pd
sample = pd.read_parquet("data/processed/gold_sentiment_sample.parquet")
print(sample[sample["rating"] == 1][["review_text", "sentiment_label", "sentiment_score"]].sample(5))
