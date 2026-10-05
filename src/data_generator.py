"""Generate synthetic product + interaction data."""
import os
import numpy as np
import pandas as pd


PRODUCT_CATEGORIES = {
    "Electronics": ["Laptop", "Smartphone", "Headphones", "Camera", "Tablet", "Smartwatch", "Speaker", "Monitor"],
    "Books": ["Fiction Novel", "Science Book", "History Book", "Self Help", "Biography", "Cookbook", "Textbook"],
    "Clothing": ["T-Shirt", "Jeans", "Jacket", "Sneakers", "Dress", "Hoodie", "Formal Shirt"],
    "Home": ["Coffee Maker", "Blender", "Vacuum Cleaner", "Air Purifier", "Lamp", "Chair", "Desk"],
    "Sports": ["Yoga Mat", "Dumbbells", "Running Shoes", "Bicycle", "Tennis Racket", "Basketball", "Gym Bag"],
}

BRANDS = ["Nova", "Zenith", "Apex", "Lumina", "Vertex", "Pulse", "Nimbus", "Orbit"]
ADJECTIVES = ["Premium", "Compact", "Wireless", "Portable", "Smart", "Ultra", "Pro", "Essential"]


def generate_products(n_products: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic product catalog."""
    np.random.seed(seed)
    rows = []
    categories = list(PRODUCT_CATEGORIES.keys())

    for pid in range(1, n_products + 1):
        category = np.random.choice(categories)
        subcategory = np.random.choice(PRODUCT_CATEGORIES[category])
        brand = np.random.choice(BRANDS)
        adj = np.random.choice(ADJECTIVES)

        title = f"{brand} {adj} {subcategory}"
        desc = (
            f"{title} - a high-quality {subcategory.lower()} in the "
            f"{category.lower()} category. Features include durability, "
            f"modern design, and excellent value."
        )

        rows.append({
            "product_id": pid,
            "title": title,
            "description": desc,
            "category": category,
            "subcategory": subcategory,
            "brand": brand,
            "price": round(np.random.uniform(10, 1500), 2),
        })

    return pd.DataFrame(rows)


def generate_interactions(products: pd.DataFrame, n_users: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic user-product interactions."""
    np.random.seed(seed)
    n_products = len(products)

    # Each user prefers 1-2 categories
    user_prefs = {}
    all_categories = products["category"].unique()
    for uid in range(1, n_users + 1):
        prefs = np.random.choice(all_categories, size=np.random.randint(1, 3), replace=False)
        user_prefs[uid] = set(prefs)

    rows = []
    for uid in range(1, n_users + 1):
        n_interactions = np.random.randint(5, 40)
        # Bias selection toward preferred categories
        weights = products["category"].apply(lambda c: 3.0 if c in user_prefs[uid] else 1.0).values
        weights = weights / weights.sum()

        chosen = np.random.choice(n_products, size=n_interactions, replace=False, p=weights)
        for idx in chosen:
            rows.append({
                "user_id": uid,
                "product_id": products.iloc[idx]["product_id"],
                "rating": np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.10, 0.20, 0.35, 0.30]),
            })

    return pd.DataFrame(rows)


def generate_dataset(data_dir: str = "data"):
    """Generate and save both products and interactions."""
    os.makedirs(data_dir, exist_ok=True)

    products = generate_products()
    interactions = generate_interactions(products)

    products.to_csv(f"{data_dir}/products.csv", index=False)
    interactions.to_csv(f"{data_dir}/interactions.csv", index=False)

    print(f"[OK] Products: {len(products)} saved to {data_dir}/products.csv")
    print(f"[OK] Interactions: {len(interactions)} saved to {data_dir}/interactions.csv")

    return products, interactions


if __name__ == "__main__":
    products, interactions = generate_dataset()
    print(f"\nSample products:")
    print(products.head())
    print(f"\nSample interactions:")
    print(interactions.head())
