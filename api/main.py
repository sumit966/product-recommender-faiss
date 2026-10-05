"""FastAPI inference service for product recommendations."""
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

app = FastAPI(
    title="Product Recommender API",
    description="Hybrid recommender using collaborative + embedding similarity",
    version="1.0.0",
)

PRODUCTS_DF: Optional[pd.DataFrame] = None
MODEL = None


class RecommendRequest(BaseModel):
    user_rated_items: List[int] = Field(..., description="List of product IDs the user has interacted with")
    top_k: int = Field(10, ge=1, le=50)


class ProductInfo(BaseModel):
    product_id: int
    title: str
    category: str
    brand: str
    price: float


class RecommendResponse(BaseModel):
    recommendations: List[ProductInfo]


class SimilarRequest(BaseModel):
    product_id: int
    top_k: int = Field(5, ge=1, le=20)


@app.on_event("startup")
def load_model():
    global MODEL, PRODUCTS_DF
    try:
        MODEL = joblib.load("models/hybrid_recommender.joblib")
        PRODUCTS_DF = pd.read_csv("data/products.csv").set_index("product_id")
        print("[OK] Recommender loaded")
    except FileNotFoundError:
        print("[WARN] Model not found. Run: python src/train.py")


def to_product_info(pid: int) -> Optional[ProductInfo]:
    if PRODUCTS_DF is None or pid not in PRODUCTS_DF.index:
        return None
    row = PRODUCTS_DF.loc[pid]
    return ProductInfo(
        product_id=int(pid),
        title=str(row["title"]),
        category=str(row["category"]),
        brand=str(row["brand"]),
        price=float(row["price"]),
    )


@app.get("/")
def root():
    return {"message": "Product Recommender API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": MODEL is not None}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    pids = MODEL.recommend(req.user_rated_items, top_k=req.top_k)
    products = [to_product_info(p) for p in pids]
    products = [p for p in products if p is not None]
    return RecommendResponse(recommendations=products)


@app.post("/similar", response_model=RecommendResponse)
def similar(req: SimilarRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    pids = MODEL.embed.similar_to_product(req.product_id, top_k=req.top_k)
    products = [to_product_info(p) for p in pids]
    products = [p for p in products if p is not None]
    return RecommendResponse(recommendations=products)


@app.get("/product/{product_id}", response_model=ProductInfo)
def get_product(product_id: int):
    p = to_product_info(product_id)
    if p is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return p
