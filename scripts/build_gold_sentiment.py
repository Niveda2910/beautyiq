import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time

PROCESSED_DIR = "data/processed"

# --- Load Silver ---
reviews = pd.read_parquet(f"{PROCESSED_DIR}/silver_reviews.parquet")
print(f"Silver reviews: {reviews.shape}")

# --- Stratified sample (proportional across primary_category; currently all Skincare) ---
SAMPLE_SIZE = 50000
sample_frac = min(1.0, SAMPLE_SIZE / len(reviews))
sample = reviews.groupby("primary_category", group_keys=False).sample(
    frac=sample_frac, random_state=42
)
print(f"Sampled {len(sample)} rows")

# --- VADER sentiment scoring ---
analyzer = SentimentIntensityAnalyzer()

def score_text(text):
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "POSITIVE"
    elif compound <= -0.05:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"
    return label, compound

start = time.time()
results = sample["review_text"].apply(score_text)
sample["sentiment_label"] = results.apply(lambda x: x[0])
sample["sentiment_score"] = results.apply(lambda x: x[1])
print(f"Scored {len(sample)} reviews in {time.time() - start:.1f}s")

print(sample["sentiment_label"].value_counts())

# --- Sanity check: does sentiment roughly track star rating? ---
print("\nMean compound score by star rating:")
print(sample.groupby("rating")["sentiment_score"].mean())

# --- Save Gold ---
sample.to_parquet(f"{PROCESSED_DIR}/gold_sentiment_sample.parquet", index=False)
print("Saved gold_sentiment_sample.parquet")