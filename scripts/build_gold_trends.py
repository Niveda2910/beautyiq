import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

PROCESSED_DIR = "data/processed"

# --- Load the sentiment-scored Gold sample (reuse it, add trend layer on top) ---
df = pd.read_parquet(f"{PROCESSED_DIR}/gold_sentiment_sample.parquet")
print(f"Loaded: {df.shape}")

df["submission_month"] = df["submission_time"].dt.to_period("M").astype(str)

# =========================================================
# PART 1: Ingredient/keyword trend tracking
# =========================================================
INGREDIENTS = [
    "retinol", "niacinamide", "hyaluronic acid", "vitamin c", "salicylic acid",
    "peptides", "ceramides", "azelaic acid", "snail mucin", "bakuchiol",
    "squalane", "glycolic acid", "spf", "collagen"
]

def find_ingredients(text, ingredient_list):
    text_lower = text.lower()
    return [ing for ing in ingredient_list if ing in text_lower]

df["ingredients_mentioned"] = df["review_text"].apply(lambda t: find_ingredients(t, INGREDIENTS))

# Explode into one row per (review, ingredient) mention
ingredient_rows = df[["submission_month", "ingredients_mentioned", "sentiment_score"]].explode("ingredients_mentioned")
ingredient_rows = ingredient_rows.dropna(subset=["ingredients_mentioned"])

trend_table = (
    ingredient_rows.groupby(["submission_month", "ingredients_mentioned"])
    .agg(mention_count=("ingredients_mentioned", "count"), avg_sentiment=("sentiment_score", "mean"))
    .reset_index()
    .rename(columns={"ingredients_mentioned": "ingredient"})
)

print("\nTop ingredients by total mentions:")
print(ingredient_rows["ingredients_mentioned"].value_counts().head(10))

trend_table.to_parquet(f"{PROCESSED_DIR}/gold_ingredient_trends.parquet", index=False)
print("Saved gold_ingredient_trends.parquet")

# =========================================================
# PART 2: Data-driven topic modeling (TF-IDF + NMF)
# =========================================================
N_TOPICS = 10

vectorizer = TfidfVectorizer(
    max_df=0.9, min_df=10, stop_words="english", ngram_range=(1, 2), max_features=5000
)
tfidf = vectorizer.fit_transform(df["review_text"])

nmf = NMF(n_components=N_TOPICS, random_state=42, max_iter=300)
topic_weights = nmf.fit_transform(tfidf)
df["topic_id"] = topic_weights.argmax(axis=1)

feature_names = vectorizer.get_feature_names_out()
print("\nTop words per discovered topic:")
topic_labels = {}
for idx, topic in enumerate(nmf.components_):
    top_words = [feature_names[i] for i in topic.argsort()[-8:][::-1]]
    topic_labels[idx] = ", ".join(top_words)
    print(f"Topic {idx}: {topic_labels[idx]}")

df["topic_label"] = df["topic_id"].map(topic_labels)

topic_trend = (
    df.groupby(["submission_month", "topic_id", "topic_label"])
    .agg(review_count=("topic_id", "count"), avg_sentiment=("sentiment_score", "mean"))
    .reset_index()
)

topic_trend.to_parquet(f"{PROCESSED_DIR}/gold_topic_trends.parquet", index=False)
df.to_parquet(f"{PROCESSED_DIR}/gold_reviews_with_topics.parquet", index=False)
print("Saved gold_topic_trends.parquet and gold_reviews_with_topics.parquet")