import pandas as pd
import os

PROCESSED_DIR = "data/processed"

# --- Load Bronze ---
reviews = pd.read_parquet(f"{PROCESSED_DIR}/bronze_reviews.parquet")
products = pd.read_parquet(f"{PROCESSED_DIR}/bronze_products.parquet")

print(f"Bronze reviews: {reviews.shape}")
print(f"Bronze products: {products.shape}")

# --- 1. Drop rows with missing review text ---
reviews = reviews.dropna(subset=["review_text"])
print(f"After dropping missing review_text: {reviews.shape}")

# --- 2. Drop duplicate reviews ---
reviews = reviews.drop_duplicates(subset=["author_id", "product_id", "submission_time"])
print(f"After dropping duplicates: {reviews.shape}")

# --- 3. Normalize timestamp ---
reviews["submission_time"] = pd.to_datetime(reviews["submission_time"], errors="coerce")
bad_dates = reviews["submission_time"].isna().sum()
print(f"Rows with unparseable submission_time: {bad_dates}")
reviews = reviews.dropna(subset=["submission_time"])

# --- 4. Filter out very short/low-signal reviews ---
reviews["review_text_len"] = reviews["review_text"].str.len()
before = reviews.shape[0]
reviews = reviews[reviews["review_text_len"] >= 10]
print(f"Dropped {before - reviews.shape[0]} reviews under 10 characters")

# --- 5. Clean text: strip whitespace, normalize ---
reviews["review_text"] = reviews["review_text"].str.strip()
reviews["review_title"] = reviews["review_title"].fillna("").str.strip()

# --- 6. Join to product category info ---
products_slim = products[[
    "product_id", "primary_category", "secondary_category",
    "tertiary_category", "brand_name", "price_usd"
]].rename(columns={"brand_name": "product_brand_name", "price_usd": "product_price_usd"})

reviews_enriched = reviews.merge(products_slim, on="product_id", how="left")

missing_category = reviews_enriched["primary_category"].isna().sum()
print(f"Reviews with no matching product (orphaned product_id): {missing_category}")

# --- 7. Final column selection for Silver ---
silver_cols = [
    "author_id", "product_id", "product_name", "brand_name",
    "primary_category", "secondary_category", "tertiary_category",
    "rating", "is_recommended", "review_title", "review_text",
    "review_text_len", "skin_type", "skin_tone", "eye_color", "hair_color",
    "submission_time", "helpfulness", "total_feedback_count",
    "product_price_usd"
]
reviews_silver = reviews_enriched[silver_cols]

print(f"\nFinal Silver reviews shape: {reviews_silver.shape}")

# --- Save ---
reviews_silver.to_parquet(f"{PROCESSED_DIR}/silver_reviews.parquet", index=False)
print("Saved silver_reviews.parquet")