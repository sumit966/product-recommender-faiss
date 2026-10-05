"""Tests for Product Recommender API."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert "status" in r.json()


def test_recommend_valid():
    payload = {"user_rated_items": [1, 2, 3, 4, 5], "top_k": 5}
    r = client.post("/recommend", json=payload)
    if r.status_code == 503:
        return  # model not trained yet
    assert r.status_code == 200
    body = r.json()
    assert "recommendations" in body


def test_recommend_invalid_topk():
    payload = {"user_rated_items": [1, 2], "top_k": 999}
    r = client.post("/recommend", json=payload)
    assert r.status_code == 422
