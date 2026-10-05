"""Hybrid recommender combining collaborative + embedding similarity."""
import numpy as np
import pandas as pd

from collaborative import CollaborativeRecommender
from embeddings import EmbeddingRecommender


class HybridRecommender:
    """Weighted blend of collaborative and embedding recommendations."""

    def __init__(self, alpha: float = 0.5):
        """
        alpha = weight for embedding score
        (1 - alpha) = weight for collaborative score
        """
        self.alpha = alpha
        self.collab = CollaborativeRecommender()
        self.embed = EmbeddingRecommender()

    def fit(self, products: pd.DataFrame, interactions: pd.DataFrame):
        self.collab.fit(interactions)
        self.embed.fit(products)
        print(f"[OK] Hybrid recommender ready (alpha={self.alpha})")
        return self

    def recommend(self, user_rated_items: list, top_k: int = 10) -> list:
        """Recommend combining both signals."""
        collab_recs = self.collab.recommend_from_rated(user_rated_items, top_k * 3)
        embed_recs = []
        if user_rated_items:
            # Aggregate embedding similarity for each rated item
            for pid in user_rated_items[-3:]:  # last 3 items
                embed_recs.extend(self.embed.similar_to_product(pid, top_k * 2))

        # Weighted scoring
        scores = {}
        for rank, pid in enumerate(collab_recs):
            scores[pid] = scores.get(pid, 0) + (1 - self.alpha) * (1.0 / (rank + 1))

        for rank, pid in enumerate(embed_recs):
            scores[pid] = scores.get(pid, 0) + self.alpha * (1.0 / (rank + 1))

        # Exclude already-rated
        for pid in user_rated_items:
            scores.pop(pid, None)

        return [pid for pid, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]]
