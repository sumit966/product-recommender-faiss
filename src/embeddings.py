"""Embedding-based recommender using Sentence Transformers.

Falls back to TF-IDF if sentence-transformers or faiss is not installed
(e.g., on Python 3.14 where wheels may not exist yet).
"""
import os
import numpy as np
import pandas as pd
import joblib


class EmbeddingRecommender:
    """Recommend products by embedding similarity.

    Uses Sentence Transformers + FAISS when available.
    Falls back to TF-IDF + cosine similarity otherwise.
    """

    def __init__(self):
        self.mode = None  # "faiss" or "tfidf"
        self.product_ids = None
        self.embeddings = None
        self.vectorizer = None
        self.model = None
        self.index = None
        self._texts = None

    def _try_load_st(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            return True
        except Exception as e:
            print(f"[WARN] Sentence Transformers unavailable ({e}). Falling back to TF-IDF.")
            return False

    def _try_load_faiss(self):
        try:
            import faiss
            return faiss
        except Exception as e:
            print(f"[WARN] FAISS unavailable ({e}). Using numpy cosine similarity.")
            return None

    def fit(self, products: pd.DataFrame):
        """Build product text embeddings."""
        self.product_ids = products["product_id"].values
        self._texts = (products["title"] + " " + products["description"]).tolist()

        faiss = self._try_load_faiss()
        use_st = self._try_load_st()

        if use_st and faiss is not None:
            self.mode = "faiss"
            embeddings = self.model.encode(
                self._texts, show_progress_bar=False, convert_to_numpy=True
            ).astype("float32")
            # L2-normalize for cosine
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, 1e-8, None)
            self.embeddings = embeddings

            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(embeddings)
            print(f"[OK] Embedding model: Sentence Transformers + FAISS ({embeddings.shape})")
        else:
            self.mode = "tfidf"
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(max_features=512, stop_words="english")
            self.embeddings = self.vectorizer.fit_transform(self._texts).toarray().astype("float32")
            # L2 normalize
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            self.embeddings = self.embeddings / np.clip(norms, 1e-8, None)
            print(f"[OK] Embedding model: TF-IDF fallback ({self.embeddings.shape})")

        return self

    def similar_to_text(self, query: str, top_k: int = 10) -> list:
        """Find products similar to a text query."""
        if self.mode == "faiss":
            q = self.model.encode([query], convert_to_numpy=True).astype("float32")
            norms = np.linalg.norm(q, axis=1, keepdims=True)
            q = q / np.clip(norms, 1e-8, None)
            _, idx = self.index.search(q, top_k)
            return [int(self.product_ids[i]) for i in idx[0]]
        elif self.mode == "tfidf":
            q = self.vectorizer.transform([query]).toarray().astype("float32")
            norms = np.linalg.norm(q, axis=1, keepdims=True)
            q = q / np.clip(norms, 1e-8, None)
            scores = self.embeddings @ q.T
            scores = scores.flatten()
            top_idx = np.argsort(scores)[-top_k:][::-1]
            return [int(self.product_ids[i]) for i in top_idx]
        return []

    def similar_to_product(self, product_id: int, top_k: int = 10) -> list:
        """Find products similar to a given product."""
        pid_to_idx = {pid: i for i, pid in enumerate(self.product_ids)}
        if product_id not in pid_to_idx:
            return []
        idx = pid_to_idx[product_id]

        if self.mode == "faiss":
            q = self.embeddings[idx:idx+1]
            _, idxs = self.index.search(q, top_k + 1)
            return [int(self.product_ids[i]) for i in idxs[0] if i != idx][:top_k]
        else:
            scores = self.embeddings @ self.embeddings[idx]
            scores[idx] = -np.inf
            top_idx = np.argsort(scores)[-top_k:][::-1]
            return [int(self.product_ids[i]) for i in top_idx]

    def save(self, path: str = "models"):
        os.makedirs(path, exist_ok=True)
        joblib.dump({
            "mode": self.mode,
            "product_ids": self.product_ids,
            "embeddings": self.embeddings,
            "vectorizer": self.vectorizer,
        }, f"{path}/embedding_model.joblib")
        print(f"[OK] Saved embedding model to {path}/embedding_model.joblib")
