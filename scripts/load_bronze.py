import pandas as pd
import glob
import os

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# --- Load and concatenate all review chunks ---
review_files = sorted(glob.glob(f"{RAW_DIR}/reviews_*.csv"))
print(f"Found {len(review_files)} review files: {review_files}")

review_dfs = []
for f in review_files:
    df = pd.read_csv(f, dtype={"author_id": str}, low_memory=False)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df["source_file"] = os.path.basename(f)
    review_dfs.append(df)

reviews = pd.concat(review_dfs, ignore_index=True)
print(f"Total reviews combined: {reviews.shape}")

# --- Load product info ---
products = pd.read_csv(f"{RAW_DIR}/product_info.csv")
print(f"Total products: {products.shape}")

# --- Basic sanity checks ---
print("\nMissing review_text:", reviews["review_text"].isna().sum())
print("Duplicate reviews (author_id + product_id + submission_time):",
      reviews.duplicated(subset=["author_id", "product_id", "submission_time"]).sum())

# --- Save as Bronze (local parquet for now, Databricks load next) ---
reviews.to_parquet(f"{PROCESSED_DIR}/bronze_reviews.parquet", index=False)
products.to_parquet(f"{PROCESSED_DIR}/bronze_products.parquet", index=False)

print("\nSaved bronze_reviews.parquet and bronze_products.parquet to data/processed/")