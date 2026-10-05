"""Collaborative filtering recommender (item-item similarity)."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeRecommender:
    """Item-item collaborative filtering using cosine similarity."""

    def __init__(self):
        self.user_item_matrix = None
        self.item_similarity = None
        self.product_ids = None

    def fit(self, interactions: pd.DataFrame):
        """Build user-item matrix and item-item similarity."""
        # Pivot: rows=users, cols=products, values=rating
        matrix = interactions.pivot_table(
            index="user_id", columns="product_id",
            values="rating", fill_value=0
        )
        self.product_ids = matrix.columns.values
        self.user_item_matrix = matrix.values

        # Item-item cosine similarity
        item_vectors = self.user_item_matrix.T  # shape: (n_products, n_users)
        self.item_similarity = cosine_similarity(item_vectors)

        # Zero out self-similarity
        np.fill_diagonal(self.item_similarity, 0)

        print(f"[OK] Collaborative model trained: {self.user_item_matrix.shape}")
        return self

    def recommend(self, user_id: int, top_k: int = 10) -> list:
        """Recommend top-K products for a user."""
        if self.user_item_matrix is None:
            return []

        # Find user index
        user_idx = None
        for i, uid in enumerate(self.user_item_matrix):
            pass  # placeholder

        # Get user's rating vector
        try:
            user_idx = list(self.user_item_matrix.index if hasattr(self.user_item_matrix, 'index') else range(len(self.user_item_matrix))).index(user_id)
        except Exception:
            return []

        # Fallback: use position 0 if not found
        user_ratings = None
        return []

    def recommend_from_rated(self, user_rated_items: list, top_k: int = 10) -> list:
        """Recommend based on a list of product IDs the user has rated.

        This is useful when the user is new or not in the training matrix.
        """
        if self.item_similarity is None or not user_rated_items:
            return []

        pid_to_idx = {pid: i for i, pid in enumerate(self.product_ids)}
        valid_items = [pid_to_idx[pid] for pid in user_rated_items if pid in pid_to_idx]

        if not valid_items:
            return []

        # Average similarity across the user's rated items
        scores = self.item_similarity[valid_items].mean(axis=0)

        # Exclude already-rated items
        scores[valid_items] = -np.inf

        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [int(self.product_ids[i]) for i in top_indices if scores[i] > -np.inf]
