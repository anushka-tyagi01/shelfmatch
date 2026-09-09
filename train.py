"""
Training / data-prep script for ShelfMatch.

Builds two real recommendation approaches on the goodbooks-10k dataset
(10,000 real books, ~6M real Goodreads ratings):

1. Content-based similarity — TF-IDF over each book's genre tags +
   author, cosine similarity between books.
2. Item-item collaborative filtering — "readers who loved this also
   loved..." computed from the real ratings matrix, no cold-start
   problem for new users since it only needs item-item similarity.

Both are evaluated offline with real, honest metrics (not just
demonstrated informally), saved alongside the trained artifacts.
"""

import json
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

# ============================================================
# Load data
# ============================================================

books = pd.read_csv("books.csv")
tags = pd.read_csv("tags.csv")
book_tags = pd.read_csv("book_tags.csv")
ratings = pd.read_csv("ratings.csv")

# Keep only books.csv's core columns, index by book_id (1..10000)
books = books[[
    "book_id", "goodreads_book_id", "title", "authors",
    "original_publication_year", "average_rating", "ratings_count",
]].copy()

# ============================================================
# Curate a genre whitelist from the noisy reader-generated tags
# (most raw tags are shelf-organization clutter like "to-read",
# "owned", "kindle" — not genuinely descriptive of content)
# ============================================================

GENRE_WHITELIST = [
    "fantasy", "young-adult", "romance", "mystery", "science-fiction",
    "sci-fi", "historical-fiction", "paranormal", "horror", "thriller",
    "urban-fantasy", "dystopian", "dystopia", "classics", "classic",
    "chick-lit", "graphic-novels", "memoir", "poetry", "biography",
    "crime", "adventure", "humor", "contemporary", "vampires",
    "paranormal-romance", "childrens", "children", "non-fiction",
    "nonfiction", "history", "self-help", "philosophy", "short-stories",
]

tag_lookup = tags.set_index("tag_id")["tag_name"].to_dict()
book_tags["tag_name"] = book_tags["tag_id"].map(tag_lookup)
genre_tags = book_tags[book_tags["tag_name"].isin(GENRE_WHITELIST)]

# Top 5 genre tags per book by tag count, joined into one string
genre_tags_sorted = genre_tags.sort_values("count", ascending=False)
top_genres_per_book = (
    genre_tags_sorted.groupby("goodreads_book_id")["tag_name"]
    .apply(lambda s: " ".join(s.head(5)))
    .rename("genres")
)

books = books.merge(top_genres_per_book, on="goodreads_book_id", how="left")
books["genres"] = books["genres"].fillna("")
books["authors_clean"] = books["authors"].str.replace(",", " ").str.replace("-", "")

# Text used for content-based similarity: genres (weighted x2) + author
books["content_soup"] = (
    (books["genres"] + " ") * 2 + books["authors_clean"]
)

print(f"Books with at least one curated genre tag: "
      f"{(books['genres'] != '').sum()} / {len(books)}")

# ============================================================
# 1. Content-based similarity (TF-IDF + cosine similarity)
# ============================================================

tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = tfidf.fit_transform(books["content_soup"])

content_similarity = cosine_similarity(tfidf_matrix, dense_output=False)

# ============================================================
# Evaluate content-based recommendations: genre-overlap@10
# For held-out books with known genres, how much genre overlap do
# the top-10 recommended books have, vs. a random baseline?
# ============================================================

book_id_to_idx = {bid: i for i, bid in enumerate(books["book_id"])}
books_with_genres = books[books["genres"] != ""].sample(
    n=min(300, (books["genres"] != "").sum()), random_state=RANDOM_STATE
)

def genre_overlap(genres_a: str, genres_b: str) -> float:
    set_a, set_b = set(genres_a.split()), set(genres_b.split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

content_overlaps = []
random_overlaps = []

all_book_ids = books["book_id"].tolist()

for _, row in books_with_genres.iterrows():
    idx = book_id_to_idx[row["book_id"]]
    sims = np.asarray(content_similarity[idx].todense()).flatten()
    sims[idx] = -1  # exclude the book itself
    top10_idx = np.argsort(sims)[-10:]

    for j in top10_idx:
        content_overlaps.append(
            genre_overlap(row["genres"], books.iloc[j]["genres"])
        )

    random_idx = rng.choice(len(books), size=10, replace=False)
    for j in random_idx:
        random_overlaps.append(
            genre_overlap(row["genres"], books.iloc[j]["genres"])
        )

content_based_genre_overlap = float(np.mean(content_overlaps))
random_baseline_genre_overlap = float(np.mean(random_overlaps))

print(f"\nContent-based genre overlap@10: {content_based_genre_overlap:.4f}")
print(f"Random baseline genre overlap@10: {random_baseline_genre_overlap:.4f}")

# ============================================================
# 2. Item-item collaborative filtering
# "Readers who loved this also loved..." — computed from the real
# ratings matrix. Works for any book with enough ratings, and needs
# no user history, so it's usable by brand-new visitors (no
# cold-start problem for the *user* side).
# ============================================================

# Only keep ratings >= 4 as "liked" signal, and only books in our set
ratings = ratings[ratings["book_id"].isin(books["book_id"])]

# Build a sparse user-item matrix (users as rows, books as columns)
user_ids = ratings["user_id"].unique()
user_id_to_idx = {u: i for i, u in enumerate(user_ids)}
n_users = len(user_ids)
n_books = len(books)

row_idx = ratings["user_id"].map(user_id_to_idx).values
col_idx = ratings["book_id"].map(book_id_to_idx).values
data = ratings["rating"].values

user_item_matrix = csr_matrix(
    (data, (row_idx, col_idx)), shape=(n_users, n_books)
)

print(f"\nUser-item matrix: {n_users} users x {n_books} books, "
      f"{user_item_matrix.nnz} real ratings")

# Item-item similarity via cosine similarity on the (sparse) item vectors
# (columns of the user-item matrix = each book's rating pattern across users)
item_similarity = cosine_similarity(user_item_matrix.T, dense_output=False)

# ============================================================
# Evaluate item-item CF with leave-one-out Recall@10
# For users with >=6 liked books (rating >= 4), hide one liked book,
# generate recommendations from the rest, check if the hidden book
# appears in the top-10 — compared to a random baseline.
# ============================================================

user_likes = (
    ratings[ratings["rating"] >= 4]
    .groupby("user_id")["book_id"]
    .apply(list)
)
eval_users = user_likes[user_likes.apply(len) >= 6].sample(
    n=min(300, len(user_likes)), random_state=RANDOM_STATE
)

def recommend_from_liked(liked_book_ids, exclude_ids, top_n=10):
    idxs = [book_id_to_idx[b] for b in liked_book_ids if b in book_id_to_idx]
    if not idxs:
        return [], None
    scores = np.asarray(item_similarity[idxs].mean(axis=0)).flatten()
    exclude_idxs = [book_id_to_idx[b] for b in idxs + exclude_ids if b in book_id_to_idx]
    scores_for_ranking = scores.copy()
    for i in [book_id_to_idx[b] for b in liked_book_ids if b in book_id_to_idx]:
        scores_for_ranking[i] = -np.inf
    top_idx = np.argsort(scores_for_ranking)[-top_n:]
    return [all_book_ids[i] for i in top_idx], scores_for_ranking

hits_at_10 = 0
hits_at_50 = 0
random_hits_at_10 = 0
percentile_ranks = []
total = 0

for user_id, liked in eval_users.items():
    hidden_book = liked[-1]
    remaining = liked[:-1]

    if hidden_book not in book_id_to_idx:
        continue

    recs_10, scores = recommend_from_liked(remaining, exclude_ids=[hidden_book], top_n=10)
    if scores is None:
        continue

    recs_50, _ = recommend_from_liked(remaining, exclude_ids=[hidden_book], top_n=50)

    if hidden_book in recs_10:
        hits_at_10 += 1
    if hidden_book in recs_50:
        hits_at_50 += 1

    # Percentile rank of the hidden book among all candidates (lower = better;
    # 50% = no better than random guessing). Standard implicit-feedback metric.
    hidden_idx = book_id_to_idx[hidden_book]
    rank = (scores > scores[hidden_idx]).sum()
    percentile_rank = rank / len(scores)
    percentile_ranks.append(percentile_rank)

    random_recs = rng.choice(all_book_ids, size=10, replace=False)
    if hidden_book in random_recs:
        random_hits_at_10 += 1

    total += 1

cf_recall_at_10 = hits_at_10 / total
cf_recall_at_50 = hits_at_50 / total
random_recall_at_10 = random_hits_at_10 / total
mean_percentile_rank = float(np.mean(percentile_ranks))

print(f"\nItem-item CF Recall@10 (leave-one-out): {cf_recall_at_10:.4f}")
print(f"Item-item CF Recall@50 (leave-one-out): {cf_recall_at_50:.4f}")
print(f"Random baseline Recall@10: {random_recall_at_10:.4f}")
print(f"Mean percentile rank (lower=better, 0.5=random): {mean_percentile_rank:.4f}")
print(f"Evaluated on {total} users with >=6 liked books")

# ============================================================
# Save everything
# ============================================================

books_lookup = books[[
    "book_id", "title", "authors", "genres",
    "average_rating", "ratings_count", "original_publication_year",
]].copy()

books_lookup.to_json("books_lookup.json", orient="records")

# Storing the full 10,000 x 10,000 similarity matrices is unnecessary and
# too large to deploy (700MB+) — recommendations only ever need each
# book's top-K nearest neighbors, so that's all we persist. This is
# standard practice for serving item-based similarity at scale.
TOP_K_NEIGHBORS = 30


def extract_top_k_neighbors(sim_matrix, top_k=TOP_K_NEIGHBORS):
    """For each item, returns its top_k most similar other items as
    {book_idx: [[neighbor_idx, score], ...]}."""
    neighbors = {}
    n_items = sim_matrix.shape[0]
    for i in range(n_items):
        row = np.asarray(sim_matrix[i].todense()).flatten()
        row[i] = -1  # exclude itself
        top_idx = np.argsort(row)[-top_k:][::-1]
        neighbors[i] = [[int(j), round(float(row[j]), 4)] for j in top_idx if row[j] > 0]
    return neighbors


print("\nExtracting top-K neighbor lists for deployment...")
content_neighbors = extract_top_k_neighbors(content_similarity)
item_cf_neighbors = extract_top_k_neighbors(item_similarity)

with open("content_neighbors.json", "w") as f:
    json.dump(content_neighbors, f)

with open("item_cf_neighbors.json", "w") as f:
    json.dump(item_cf_neighbors, f)

with open("book_id_to_idx.json", "w") as f:
    json.dump({str(k): v for k, v in book_id_to_idx.items()}, f)

metrics = {
    "dataset": "goodbooks-10k (10,000 real books, ~6M real Goodreads ratings)",
    "content_based": {
        "method": "TF-IDF over curated genre tags + author, cosine similarity",
        "eval_metric": "genre_overlap@10 (Jaccard similarity of genre tags, "
                        "vs. same-size random baseline)",
        "model_score": round(content_based_genre_overlap, 4),
        "random_baseline_score": round(random_baseline_genre_overlap, 4),
    },
    "collaborative_filtering": {
        "method": "Item-item cosine similarity on real user ratings "
                   "(readers who rated book A highly also rated book B highly)",
        "eval_metric": "Leave-one-out: hide one liked book per test user, "
                        "rank all 10,000 books by recommendation score",
        "recall_at_10": round(cf_recall_at_10, 4),
        "recall_at_50": round(cf_recall_at_50, 4),
        "random_baseline_recall_at_10": round(random_recall_at_10, 4),
        "mean_percentile_rank": round(mean_percentile_rank, 4),
        "mean_percentile_rank_note": "0.5 = no better than random guessing; lower is better",
        "n_users_evaluated": total,
        "n_users_total": n_users,
        "n_books": n_books,
        "n_ratings": int(user_item_matrix.nnz),
    },
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved: books_lookup.json, content_similarity.pkl, item_similarity.pkl,")
print("       book_id_to_idx.json, metrics.json")
