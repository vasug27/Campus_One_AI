"""
Run this once locally to download the embedding model into ./models/
Then commit the ./models/ folder to GitHub so Render doesn't re-download on every boot.

Usage:
    python download_model.py
"""

from sentence_transformers import SentenceTransformer
import os

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SAVE_PATH = os.path.join(os.path.dirname(__file__), "models", "all-MiniLM-L6-v2")

os.makedirs(SAVE_PATH, exist_ok=True)

print(f"Downloading {MODEL_NAME} ...")
model = SentenceTransformer(MODEL_NAME)
model.save(SAVE_PATH)
print(f"Model saved to: {SAVE_PATH}")