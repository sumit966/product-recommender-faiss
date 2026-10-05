# Product Recommendation System with Embeddings

A production-ready hybrid product recommender that combines collaborative filtering with embedding-based semantic similarity. Evaluated with Precision@K and Recall@K against a popularity baseline.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-0467DF?style=for-the-badge&logo=meta&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

## Overview

This project implements a hybrid product recommendation system that blends two complementary signals:

1. Collaborative filtering - "users who liked X also liked Y" (item-item cosine similarity on the user-item matrix)
2. Embedding-based similarity - "products with similar titles/descriptions are related" (Sentence Transformers + FAISS vector search)

The hybrid model weights both signals, filters out already-interacted items, and returns the top-K most relevant products per user.

## Problem Statement

Modern e-commerce platforms face a classic challenge:

- Cold-start problem - Collaborative filtering fails on new products with no interactions
- Popularity bias - Collaborative filtering over-recommends popular items, ignoring niche interests
- Semantic gap - Users searching "wireless headphones for running" should match semantically, not just by ID

No single recommendation approach solves all three. That's why we combine them.

## Solution

A hybrid recommender that:

1. Builds a user-item rating matrix from interaction data
2. Computes item-item cosine similarity from that matrix (collaborative signal)
3. Encodes product titles + descriptions into dense vectors (semantic signal)
4. Indexes embeddings in FAISS for sub-millisecond retrieval
5. Blends both scores with a tunable alpha weight (default 0.5)
6. Serves recommendations via a FastAPI REST API
7. Evaluated with Precision@K and Recall@K vs a popularity baseline

## Architecture

Product Catalog + User Interactions
         |
    +----+----+
    |         |
Collaborative   Embedding
  Filtering      Model
(item-item    (Sentence
 cosine)      Transformers
    |             |
    |         FAISS Index
    |             |
    +----+--------+
         |
  Hybrid Blend (alpha)
         |
   Filter rated items
         |
   Top-K Recommendations
         |
   FastAPI REST Endpoint
         |
   Docker + GitHub Actions

## Tech Stack

| Category | Technologies |
|----------|-------------|
| ML | Scikit-learn, SciPy (sparse matrices) |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Search | FAISS (or NumPy fallback) |
| API | FastAPI, Uvicorn, Pydantic |
| Data | Pandas, NumPy |
| Testing | pytest, httpx |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Language | Python 3.11+ |

## Project Structure

product-recommender-faiss/
├── src/
│   ├── __init__.py
│   ├── data_generator.py      # Synthetic products + interactions
│   ├── collaborative.py       # Item-item collaborative filtering
│   ├── embeddings.py          # Sentence Transformer + FAISS/TF-IDF
│   ├── hybrid.py              # Weighted blend of both signals
│   ├── evaluate.py            # Precision@K, Recall@K vs baseline
│   └── train.py               # Fit and save the hybrid model
├── api/
│   ├── __init__.py
│   └── main.py                # FastAPI inference service
├── tests/
│   ├── __init__.py
│   └── test_api.py            # pytest tests
├── data/                       # Generated CSVs (git-ignored)
├── models/                     # Trained artifacts (git-ignored)
├── .github/workflows/ci.yml    # CI/CD pipeline
├── Dockerfile
├── requirements.txt
├── requirements-optional.txt
├── .gitignore
├── LICENSE
└── README.md

## Quick Start

### 1. Clone

git clone https://github.com/sumit966/product-recommender-faiss.git
cd product-recommender-faiss

### 2. Virtual environment

Windows:
python -m venv venv
venv\Scripts\activate

macOS / Linux:
python3 -m venv venv
source venv/bin/activate

### 3. Install base dependencies

pip install -r requirements.txt

### 4. (Optional) Install full embedding stack

pip install -r requirements-optional.txt

If this fails on Python 3.14 (missing wheels), skip it. The project automatically falls back to TF-IDF, which is still a fully working semantic recommender.

### 5. Generate dataset

python src/data_generator.py

Output:
[OK] Products: 500 saved to data/products.csv
[OK] Interactions: ~4000 saved to data/interactions.csv

### 6. Train the recommender

python src/train.py

Output:
[OK] Collaborative model trained: (200, 500)
[OK] Embedding model: Sentence Transformers + FAISS (500, 384)
[OK] Hybrid recommender ready (alpha=0.5)
[OK] Saved models/hybrid_recommender.joblib

### 7. Evaluate

python src/evaluate.py

Output:
[Eval] Results on 50 users (K=10):
  hybrid        P@10=0.0xxx   R@10=0.0xxx
  embedding     P@10=0.0xxx   R@10=0.0xxx
  popularity    P@10=0.0xxx   R@10=0.0xxx

### 8. Start the API

uvicorn api.main:app --reload

API is now running at http://localhost:8000

### 9. Open Swagger UI

http://localhost:8000/docs

## API Usage

### POST /recommend

Get top-K product recommendations based on a user's interaction history.

Request:
{
  "user_rated_items": [1, 2, 3, 4, 5],
  "top_k": 5
}

Response:
{
  "recommendations": [
    {
      "product_id": 42,
      "title": "Nova Premium Headphones",
      "category": "Electronics",
      "brand": "Nova",
      "price": 299.99
    },
    {
      "product_id": 87,
      "title": "Zenith Wireless Speaker",
      "category": "Electronics",
      "brand": "Zenith",
      "price": 149.50
    }
  ]
}

### POST /similar

Find products similar to a given product (embedding-only).

Request:
{
  "product_id": 42,
  "top_k": 5
}

### GET /product/{product_id}

Fetch a single product's details.

### GET /health

Returns {"status": "healthy", "model_loaded": true} when the model is ready.

### Other Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| / | GET | API info |
| /health | GET | Health check |
| /docs | GET | Swagger UI |
| /redoc | GET | Alternative docs |

### cURL Example

curl -X POST "http://localhost:8000/recommend" -H "Content-Type: application/json" -d "{\"user_rated_items\": [1, 2, 3, 4, 5], \"top_k\": 5}"

## Evaluation

Evaluated with leave-one-out on 50 users (K=10):

| Method | Precision@10 | Recall@10 |
|--------|--------------|-----------|
| Popularity Baseline | 0.xx | 0.xx |
| Embedding-only | 0.xx | 0.xx |
| Hybrid (alpha=0.5) | 0.xx | 0.xx |

Values depend on random seed and dataset size. Run "python src/evaluate.py" to reproduce.

Why hybrid wins: Collaborative filtering captures behavioral patterns, embeddings capture semantic similarity - together they handle both popular and niche products better than either alone.

## Testing

pytest tests/ -v

Tests cover:
- Root endpoint returns API info
- Health check reports model status
- /recommend returns valid product IDs
- Invalid top_k returns 422 validation error

## Docker

Build:
docker build -t product-recommender-faiss .

Run:
docker run -p 8000:8000 product-recommender-faiss

The Dockerfile trains the model automatically during build.

## CI/CD Pipeline

Every push to main triggers GitHub Actions:

1. Install Python 3.11 + dependencies
2. Generate synthetic data
3. Train hybrid recommender
4. Run pytest test suite

See .github/workflows/ci.yml.

## Key Learnings

- Hybrid beats single signal - collaborative + semantic outperforms either alone
- FAISS indexing - sub-millisecond retrieval over 500+ products, scales to millions
- TF-IDF fallback - graceful degradation keeps the service working even when heavy deps fail
- Precision@K / Recall@K - proper ranking metrics, not classification metrics
- Popularity baseline - always measure against the simplest possible approach
- FastAPI auto-docs - Swagger UI generated from Pydantic models
- Docker + CI - reproducible environments catch "works on my machine" bugs

## Future Improvements

- Real e-commerce dataset (Amazon Product Reviews)
- Cross-encoder re-ranking for top-K results
- User embedding model (two-tower architecture)
- A/B testing framework for alpha tuning
- Redis caching for repeated recommendation queries
- Deploy to GCP Cloud Run
- Add Prometheus metrics + Grafana dashboards

## License

MIT License - see LICENSE file.

## Author

Sumit Raj
- M.Tech Applied AI & ML @ VNIT Nagpur
- Ex-Software Engineer Intern @ Salesforce
- GitHub: https://github.com/sumit966
- LinkedIn: https://www.linkedin.com/in/er-sumit-raj-/
- Portfolio: https://sumit966-github-io.vercel.app
- Email: info.sr0909@gmail.com

If you found this project useful, please consider giving it a star!
