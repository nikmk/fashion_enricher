import os

BUCKET_NAME = os.getenv("BUCKET_NAME", "my-fashion-bucket")
INPUT_FILE = os.getenv("INPUT_FILE", "combined_products.json")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "enriched_products.json")
CHECKPOINT_FILE = os.getenv("CHECKPOINT_FILE", "checkpoint.txt")