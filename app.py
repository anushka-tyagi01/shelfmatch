from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import json
import os
import hashlib
import time
from datetime import datetime


# ===========================================
# Load Data & Artifacts
# ===========================================
# Trained on the real goodbooks-10k dataset (10,000 real books, ~6M
# real Goodreads ratings). See train.py and metrics.json for the
# full offline evaluation of both recommendation approaches.

MODEL_NAME = "ShelfMatch Hybrid Recommender"
MODEL_VERSION = "1.0.0"

try:
    books_df = pd.read_json("books_lookup.json")
    books_df["genres"] = books_df["genres"].fillna("")

    with open("book_id_to_idx.json") as f:
        BOOK_ID_TO_IDX = {int(k): v for k, v in json.load(f).items()}
    IDX_TO_BOOK_ID = {v: k for k, v in BOOK_ID_TO_IDX.items()}

    with open("content_neighbors.json") as f:
        CONTENT_NEIGHBORS = {int(k): v for k, v in json.load(f).items()}

    with open("item_cf_neighbors.json") as f:
        CF_NEIGHBORS = {int(k): v for k, v in json.load(f).items()}

    with open("metrics.json") as f:
        MODEL_METRICS = json.load(f)

    BOOKS_BY_IDX = books_df.set_index(
        books_df["book_id"].map(BOOK_ID_TO_IDX)
    ).to_dict(orient="index")

except Exception as e:
    raise RuntimeError(f"Error loading data/artifacts: {e}")


def title_search(query: str, limit: int = 8):
    """Simple case-insensitive substring search over book titles."""
    query_lower = query.lower()
    matches = books_df[books_df["title"].str.lower().str.contains(query_lower, na=False)]
    matches = matches.sort_values("ratings_count", ascending=False).head(limit)
    return matches[["book_id", "title", "authors"]].to_dict(orient="records")


def book_summary(book_idx: int) -> dict:
    b = BOOKS_BY_IDX[book_idx]
    return {
        "book_id": int(b["book_id"]),
        "title": b["title"],
        "authors": b["authors"],
        "genres": b["genres"],
        "average_rating": float(b["average_rating"]),
        "ratings_count": int(b["ratings_count"]),
    }


# ===========================================
# API Key Authentication
# ===========================================

API_KEY = os.environ.get("API_KEY")


def verify_api_key(x_api_key: str = Header(default=None)):
    if API_KEY is None:
        return
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Unauthorized",
                "message": "Missing or invalid API key. Provide it via the 'x-api-key' header."
            }
        )


# ===========================================
# Response Cache
# ===========================================

CACHE_TTL_SECONDS = 3600
_cache: dict[str, dict] = {}


def make_cache_key(payload: dict) -> str:
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def cache_get(key: str):
    entry = _cache.get(key)
    if entry is None:
        return None
    if time.time() - entry["stored_at"] > CACHE_TTL_SECONDS:
        del _cache[key]
        return None
    return entry["value"]


def cache_set(key: str, value: dict):
    _cache[key] = {"value": value, "stored_at": time.time()}


# ===========================================
# FastAPI
# ===========================================

app = FastAPI(
    title="ShelfMatch — Find Your Next Favorite Book 📚",
    description="""
A book recommendation API built on 10,000 real books and ~6 million
real Goodreads ratings. Combines two real recommendation techniques:

📖 **Content-based** — TF-IDF over genre tags + author, cosine similarity
👯 **Collaborative filtering** — "readers who loved this also loved..."
   computed from real rating patterns across 53,000+ real readers

Both are evaluated offline against random baselines — see /model/metrics.
""",
    version=MODEL_VERSION,
)


# ===========================================
# Schemas
# ===========================================

class SimilarBooksRequest(BaseModel):
    book_id: int = Field(description="The book_id to find similar books for")
    top_n: int = Field(default=10, ge=1, le=30)


class ForYouRequest(BaseModel):
    liked_book_ids: list[int] = Field(
        min_length=1, description="book_ids of books the reader loved"
    )
    top_n: int = Field(default=10, ge=1, le=30)


class DiscoverRequest(BaseModel):
    genres: list[str] = Field(default=[], description="Genre tags to filter by, e.g. ['fantasy', 'romance']")
    min_ratings_count: int = Field(default=1000, ge=0)
    top_n: int = Field(default=10, ge=1, le=30)


class BookResult(BaseModel):
    book_id: int
    title: str
    authors: str
    genres: str
    average_rating: float
    ratings_count: int
    score: Optional[float] = None
    reason: Optional[str] = None


class RecommendationResponse(BaseModel):
    status: str
    query: dict
    results: list[BookResult]
    cached: bool = False
    timestamp: str


# ===========================================
# Home / Health / Info
# ===========================================

@app.get("/", tags=["Home"])
def home():
    return {
        "message": "ShelfMatch — Find Your Next Favorite Book 📚",
        "status": "Running",
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "docs": "/docs",
        "health": "/health",
        "metrics": "/model/metrics",
        "n_books": len(books_df),
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "Healthy", "n_books": len(books_df)}


@app.get("/model/metrics", tags=["Information"])
def model_metrics():
    """
    Real offline evaluation of both recommendation approaches,
    each benchmarked against a random baseline. See train.py.
    """
    return MODEL_METRICS


@app.get("/cache/stats", tags=["Information"])
def cache_stats():
    return {"cached_entries": len(_cache), "ttl_seconds": CACHE_TTL_SECONDS}


@app.get("/books/search", tags=["Books"])
def search_books(q: str, limit: int = 8):
    """Search books by title (for populating a picker in the UI)."""
    if len(q) < 2:
        return {"results": []}
    return {"results": title_search(q, limit)}


# ===========================================
# Recommendation Routes
# ===========================================

@app.post(
    "/recommend/similar-books",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
    dependencies=[Depends(verify_api_key)],
)
def similar_books(payload: SimilarBooksRequest):
    """
    Content-based: "if you liked this book, you might like..."
    Uses TF-IDF similarity over genre tags + author.
    """
    try:
        cache_key = make_cache_key({"endpoint": "similar", **payload.model_dump()})
        cached = cache_get(cache_key)
        if cached is not None:
            return RecommendationResponse(**{**cached, "cached": True})

        if payload.book_id not in BOOK_ID_TO_IDX:
            raise HTTPException(status_code=404, detail="book_id not found")

        idx = BOOK_ID_TO_IDX[payload.book_id]
        neighbors = CONTENT_NEIGHBORS.get(idx, [])[: payload.top_n]

        query_book = book_summary(idx)
        results = []
        for neighbor_idx, score in neighbors:
            b = book_summary(neighbor_idx)
            shared_genres = set(b["genres"].split()) & set(query_book["genres"].split())
            reason = (
                f"Shares genres: {', '.join(shared_genres)}" if shared_genres
                else "Similar style/author"
            )
            results.append(BookResult(**b, score=round(score, 4), reason=reason))

        response = RecommendationResponse(
            status="success",
            query=query_book,
            results=results,
            timestamp=datetime.now().isoformat(),
        )
        cache_set(cache_key, response.model_dump())
        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "Recommendation failed", "message": str(e)})


@app.post(
    "/recommend/for-you",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
    dependencies=[Depends(verify_api_key)],
)
def for_you(payload: ForYouRequest):
    """
    Item-item collaborative filtering: "readers who loved these books
    also loved..." — computed from real rating patterns, works for
    any new reader with no cold-start problem (only needs a few
    books they've liked, not a pre-existing account history).
    """
    try:
        cache_key = make_cache_key({"endpoint": "for_you", **payload.model_dump()})
        cached = cache_get(cache_key)
        if cached is not None:
            return RecommendationResponse(**{**cached, "cached": True})

        liked_idxs = [BOOK_ID_TO_IDX[b] for b in payload.liked_book_ids if b in BOOK_ID_TO_IDX]
        if not liked_idxs:
            raise HTTPException(status_code=404, detail="None of the provided book_ids were found")

        # Aggregate neighbor scores across all liked books
        score_map: dict[int, float] = {}
        contributor_map: dict[int, list[str]] = {}

        for liked_idx in liked_idxs:
            liked_title = BOOKS_BY_IDX[liked_idx]["title"]
            for neighbor_idx, score in CF_NEIGHBORS.get(liked_idx, []):
                if neighbor_idx in liked_idxs:
                    continue
                score_map[neighbor_idx] = score_map.get(neighbor_idx, 0) + score
                contributor_map.setdefault(neighbor_idx, []).append(liked_title)

        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[: payload.top_n]

        results = []
        for neighbor_idx, agg_score in ranked:
            b = book_summary(neighbor_idx)
            contributors = contributor_map[neighbor_idx][:2]
            reason = f"Loved by readers who also loved: {', '.join(contributors)}"
            results.append(BookResult(**b, score=round(agg_score, 4), reason=reason))

        response = RecommendationResponse(
            status="success",
            query={"liked_book_ids": payload.liked_book_ids},
            results=results,
            timestamp=datetime.now().isoformat(),
        )
        cache_set(cache_key, response.model_dump())
        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "Recommendation failed", "message": str(e)})


@app.post(
    "/recommend/discover",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
    dependencies=[Depends(verify_api_key)],
)
def discover(payload: DiscoverRequest):
    """
    Mood/genre-based discovery, ranked by a Bayesian weighted rating
    (not a naive average) — the same technique IMDB uses, so a book
    with 5 ratings averaging 5.0 doesn't outrank one with 50,000
    ratings averaging 4.6.
    """
    try:
        cache_key = make_cache_key({"endpoint": "discover", **payload.model_dump()})
        cached = cache_get(cache_key)
        if cached is not None:
            return RecommendationResponse(**{**cached, "cached": True})

        candidates = books_df[books_df["ratings_count"] >= payload.min_ratings_count].copy()

        if payload.genres:
            genre_set = set(g.lower() for g in payload.genres)
            candidates = candidates[
                candidates["genres"].apply(
                    lambda g: bool(genre_set & set(g.lower().split()))
                )
            ]

        if candidates.empty:
            return RecommendationResponse(
                status="success",
                query=payload.model_dump(),
                results=[],
                timestamp=datetime.now().isoformat(),
            )

        # Bayesian weighted rating: WR = (v/(v+m))*R + (m/(v+m))*C
        # v = ratings_count, R = average_rating, m = minimum threshold,
        # C = mean rating across the whole candidate pool
        m = payload.min_ratings_count
        C = candidates["average_rating"].mean()
        v = candidates["ratings_count"]
        R = candidates["average_rating"]
        candidates["weighted_rating"] = (v / (v + m)) * R + (m / (v + m)) * C

        top = candidates.sort_values("weighted_rating", ascending=False).head(payload.top_n)

        results = [
            BookResult(
                book_id=int(row["book_id"]), title=row["title"], authors=row["authors"],
                genres=row["genres"], average_rating=float(row["average_rating"]),
                ratings_count=int(row["ratings_count"]),
                score=round(float(row["weighted_rating"]), 4),
                reason=f"Weighted rating (Bayesian-adjusted for {int(row['ratings_count']):,} ratings)",
            )
            for _, row in top.iterrows()
        ]

        response = RecommendationResponse(
            status="success",
            query=payload.model_dump(),
            results=results,
            timestamp=datetime.now().isoformat(),
        )
        cache_set(cache_key, response.model_dump())
        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "Discovery failed", "message": str(e)})
