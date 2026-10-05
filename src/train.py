"""Train and save hybrid recommender."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import joblib
import pandas as pd

from hybrid import HybridRecommender


def train():
    products = pd.read_csv("data/products.csv")
    interactions = pd.read_csv("data/interactions.csv")

    model = HybridRecommender(alpha=0.5).fit(products, interactions)

    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/hybrid_recommender.joblib")
    model.embed.save("models")

    print("[OK] Saved models/hybrid_recommender.joblib")


if __name__ == "__main__":
    train()
