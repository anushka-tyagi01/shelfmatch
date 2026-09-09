# 📚 ShelfMatch — Find Your Next Favorite Book

A book recommendation API built on **10,000 real books** and **~6 million real Goodreads ratings**. Combines two genuine recommendation techniques — content-based filtering and item-item collaborative filtering — both evaluated offline against random baselines, not just demonstrated informally.

**Live demo:** https://coffeeshopretailanalysis-lxh9qutmayegsfikudvhui.streamlit.app/
**API docs (Swagger):** https://shelfmatch.onrender.com

> Note: hosted on free tiers and may take 30-60 seconds to wake up after inactivity.

## What it does

Three ways to find your next book:

1. **💕 Similar Books** — content-based: "if you loved this book, you'll probably like these" (TF-IDF over genre tags + author, cosine similarity)
2. **👯 For You** — item-item collaborative filtering: "readers who loved the books you loved also loved..." — computed from real rating patterns across 53,000+ real Goodreads users. No cold-start problem: works for a brand-new visitor with just 1-3 liked books, no account history needed.
3. **🌈 Discover** — mood/genre browsing, ranked by a **Bayesian-weighted rating** (the same technique IMDB uses for its Top 250), so a book with 5 five-star ratings can't outrank one with 50,000 ratings averaging 4.6.

## Model evaluation — real numbers, not vibes

Both recommendation approaches were evaluated offline against a random baseline of the same size, using held-out real data:

### Content-based matching

| Metric | ShelfMatch | Random baseline |
|---|---|---|
| Genre overlap@10 | **79.5%** | 12.9% |

For 300 held-out books, the top-10 recommended books share 79.5% of their genre tags with the query book on average (Jaccard similarity) — over **6x better** than randomly picking 10 books.

### Collaborative filtering (item-item)

| Metric | ShelfMatch | Random baseline |
|---|---|---|
| Recall@10 (leave-one-out) | **2.33%** | 0.33% |
| Recall@50 | 16.3% | — |
| Mean percentile rank | **7.9%** | 50% (by definition) |

Evaluated via leave-one-out on 300 real readers with 6+ liked books each: hide one liked book, generate recommendations from the rest, see if the hidden book gets recovered. A mean percentile rank of 7.9% means the held-out book is ranked, on average, in the **top 8%** of all 10,000 books — vs. the 50% you'd expect from random guessing. Recall@10 in a 10,000-item catalog is a genuinely hard, high-variance metric (this is normal for recommender systems at this catalog size — that's exactly why percentile rank is reported alongside it, as a more stable, standard metric for implicit-feedback recommenders).

See `train.py` for the full pipeline and `metrics.json` for the raw numbers (also live at `/model/metrics`).

## Why two different techniques?

They solve different problems:
- **Content-based** works instantly for any single book, even ones with very few ratings (no cold-start problem for new *books*).
- **Collaborative filtering** captures patterns content-based matching can't see — e.g. readers who loved a gritty literary novel *and* a specific fantasy series, a connection no genre tag would surface — but needs real rating data to exist for the neighbor relationship to have been learned.

## Architecture

```
┌─────────────┐         ┌──────────────────────┐
│  Streamlit  │ ──────► │   FastAPI API         │
│  Frontend   │  HTTPS  │   (Render)            │
└─────────────┘         │  - /recommend/        │
                         │    similar-books      │
                         │  - /recommend/for-you │
                         │  - /recommend/discover│
                         │  - /model/metrics     │
                         └──────────────────────┘
                                  │
                     ┌────────────────────────────┐
                     │ Content similarity (TF-IDF) │
                     │ Item-item CF (real ratings) │
                     │ Top-30 neighbors per book,  │
                     │ precomputed for fast serving│
                     └────────────────────────────┘
```

**Note on storage:** the full 10,000×10,000 similarity matrices are ~700MB each — too large to deploy. Since recommendations only ever need each book's nearest neighbors, only the top-30 neighbors per book are persisted (a few MB total), which is standard practice for serving item-based similarity at scale.

## Key features

| Feature | Description |
|---|---|
| **Similar books** | `POST /recommend/similar-books` — content-based matching |
| **For you** | `POST /recommend/for-you` — real collaborative filtering |
| **Discover** | `POST /recommend/discover` — genre + Bayesian-weighted rating |
| **Book search** | `GET /books/search?q=...` — powers the picker in the UI |
| **Model metrics** | `GET /model/metrics` — live offline evaluation numbers |
| **Caching** | Identical requests return cached results |
| **API key auth** | Optional — set `API_KEY` env var to require `x-api-key` header |
| **Containerized** | Dockerized for consistent local/deployment behavior |

## Tech stack

- **Backend:** FastAPI, scikit-learn, scipy, pandas
- **Frontend:** Streamlit (custom pastel theme)
- **Deployment:** Docker, Render (API), Streamlit Community Cloud (UI)
- **Data:** [goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) — 10,000 books, ~6M ratings, real Goodreads data

## Key decisions

- **Top-K neighbor storage instead of full similarity matrices:** the full matrices were ~700MB each (undeployable); only the top 30 neighbors per book are needed for serving, cutting storage to a few MB with zero loss of recommendation quality.
- **Percentile rank alongside Recall@10:** strict top-N recall is extremely harsh at a 10,000-item catalog size and high-variance with a small eval set; percentile rank is the standard, more stable metric used in recommender-systems literature for exactly this reason.
- **Bayesian-weighted rating for "Discover," not raw average:** a naive average lets a book with 3 five-star ratings outrank a genuinely excellent book with 50,000 ratings — the weighted formula (same one IMDB uses) corrects for this.
- **Curated genre whitelist, not raw tags:** goodbooks-10k's raw reader tags are mostly shelf-organization noise (`to-read`, `owned`, `kindle`) rather than genre — a manually curated whitelist of ~30 real genre tags was used instead.

## Running locally

```bash
# Retrain from scratch (optional — all artifacts are already included).
# Note: ratings.csv (72MB) isn't tracked in git since it's only needed
# for retraining, not serving — download it separately if you want to
# rerun train.py: https://github.com/zygmuntz/goodbooks-10k
python train.py

# Backend
pip install -r requirements.txt
uvicorn app:app --reload

# Frontend (separate terminal)
streamlit run Frontend.py
```

## Verified working end-to-end

Every endpoint was tested against this exact codebase before delivery. Querying "similar books" for *The Hunger Games* correctly surfaces its own sequels and other dystopian YA titles; asking "for you" recommendations after liking *The Hunger Games* + *Harry Potter* correctly surfaces *Catching Fire* and more Harry Potter books — genuine collaborative signal recovered from real reader behavior, not hardcoded rules.

## Future improvements

- Hybrid re-ranking: blend content-based and CF scores into a single ranked list
- Matrix factorization (SVD) as a stronger CF baseline to compare against item-item similarity
- User accounts with persistent "liked books" instead of re-entering each session
- A/B test infrastructure to compare recommendation strategies on real engagement
