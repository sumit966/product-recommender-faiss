"""Evaluate recommender with Precision@K and Recall@K."""
import numpy as np
import pandas as pd

from embeddings import EmbeddingRecommender
from collaborative import CollaborativeRecommender
from hybrid import HybridRecommender


def precision_recall_at_k(recommended: list, relevant: set, k: int):
    if not recommended:
        return 0.0, 0.0
    rec_k = recommended[:k]
    hits = len(set(rec_k) & relevant)
    precision = hits / k
    recall = hits / len(relevant) if relevant else 0.0
    return precision, recall


def popularity_baseline(interactions: pd.DataFrame, top_k: int = 10) -> list:
    """Return top-K most popular products (baseline)."""
    counts = interactions.groupby("product_id").size().sort_values(ascending=False)
    return counts.head(top_k).index.tolist()


def evaluate(products: pd.DataFrame, interactions: pd.DataFrame, k: int = 10, n_users: int = 50):
    """Run leave-one-out evaluation."""
    print("\n[Eval] Training models...")
    collab = CollaborativeRecommender().fit(interactions)
    embed = EmbeddingRecommender().fit(products)
    hybrid = HybridRecommender(alpha=0.5)
    hybrid.collab = collab
    hybrid.embed = embed

    popularity = set(popularity_baseline(interactions, k))

    user_groups = interactions.groupby("user_id")
    user_ids = list(user_groups.groups.keys())[:n_users]

    metrics = {"hybrid": [], "embedding": [], "popularity": []}

    for uid in user_ids:
        user_df = user_groups.get_group(uid)
        if len(user_df) < 3:
            continue

        # Leave-one-out: last item as ground truth
        user_df_sorted = user_df.sort_values("product_id")
        held_out = int(user_df_sorted.iloc[-1]["product_id"])
        rated = user_df_sorted.iloc[:-1]["product_id"].astype(int).tolist()
        relevant = {held_out}

        # Hybrid
        recs_h = hybrid.recommend(rated, top_k=k)
        p_h, r_h = precision_recall_at_k(recs_h, relevant, k)
        metrics["hybrid"].append((p_h, r_h))

        # Embedding-only
        recs_e = []
        for pid in rated[-3:]:
            recs_e.extend(embed.similar_to_product(pid, k))
        recs_e = list(dict.fromkeys(recs_e))[:k]
        p_e, r_e = precision_recall_at_k(recs_e, relevant, k)
        metrics["embedding"].append((p_e, r_e))

        # Popularity baseline
        p_p, r_p = precision_recall_at_k(list(popularity), relevant, k)
        metrics["popularity"].append((p_p, r_p))

    print(f"\n[Eval] Results on {len(metrics['hybrid'])} users (K={k}):")
    results = {}
    for name, vals in metrics.items():
        if not vals:
            continue
        avg_p = np.mean([v[0] for v in vals])
        avg_r = np.mean([v[1] for v in vals])
        results[name] = {"precision@k": float(avg_p), "recall@k": float(avg_r)}
        print(f"  {name:12s}  P@{k}={avg_p:.4f}   R@{k}={avg_r:.4f}")

    return results


if __name__ == "__main__":
    products = pd.read_csv("data/products.csv")
    interactions = pd.read_csv("data/interactions.csv")
    evaluate(products, interactions, k=10, n_users=50)
