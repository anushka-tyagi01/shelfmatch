import streamlit as st
import requests
import os
import time

# ===========================================
# Config
# ===========================================

API_URL = "https://shelfmatch.onrender.com"
API_KEY = os.environ.get("API_KEY")

GENRE_OPTIONS = [
    "fantasy", "romance", "young-adult", "mystery", "science-fiction",
    "historical-fiction", "paranormal", "horror", "thriller", "classics",
    "contemporary", "graphic-novels", "memoir", "biography", "poetry",
]


def auth_headers() -> dict:
    if API_KEY:
        return {"x-api-key": API_KEY}
    return {}


st.set_page_config(page_title="ShelfMatch 📚✨", page_icon="📚", layout="centered")

# ---- Girly pastel styling ----
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #fff0f6 0%, #fdf2f8 40%, #fff9fb 100%);
    }
    .shelf-card {
        background: white;
        border-radius: 18px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 2px 10px rgba(255, 105, 180, 0.12);
        border: 1px solid #ffe0ec;
    }
    .shelf-title {
        font-weight: 700;
        font-size: 17px;
        color: #b5397a;
    }
    .shelf-author {
        color: #8a8a8a;
        font-size: 13px;
        margin-bottom: 6px;
    }
    .shelf-reason {
        font-size: 12.5px;
        color: #c2618d;
        font-style: italic;
    }
    .shelf-rating {
        color: #e88ab0;
        font-weight: 600;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #ff8fb1, #ffb3c6);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 8px 20px;
        font-weight: 600;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #ff6f9c, #ff9bb8);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📚 ShelfMatch")
st.caption(
    "✨ Find your next favorite read — powered by real recommendations from "
    "10,000 books and 6 million real Goodreads ratings, not just star averages."
)


# ===========================================
# Helpers
# ===========================================

def search_books(query: str):
    if len(query) < 2:
        return []
    try:
        r = requests.get(f"{API_URL}/books/search", params={"q": query}, timeout=30)
        r.raise_for_status()
        return r.json()["results"]
    except Exception:
        return []


def call_similar_books(book_id: int, top_n: int = 8):
    start = time.time()
    r = requests.post(
        f"{API_URL}/recommend/similar-books",
        json={"book_id": book_id, "top_n": top_n},
        headers=auth_headers(), timeout=60,
    )
    elapsed_ms = round((time.time() - start) * 1000, 1)
    r.raise_for_status()
    return r.json(), elapsed_ms


def call_for_you(liked_book_ids: list, top_n: int = 8):
    start = time.time()
    r = requests.post(
        f"{API_URL}/recommend/for-you",
        json={"liked_book_ids": liked_book_ids, "top_n": top_n},
        headers=auth_headers(), timeout=60,
    )
    elapsed_ms = round((time.time() - start) * 1000, 1)
    r.raise_for_status()
    return r.json(), elapsed_ms


def call_discover(genres: list, min_ratings: int, top_n: int = 10):
    start = time.time()
    r = requests.post(
        f"{API_URL}/recommend/discover",
        json={"genres": genres, "min_ratings_count": min_ratings, "top_n": top_n},
        headers=auth_headers(), timeout=60,
    )
    elapsed_ms = round((time.time() - start) * 1000, 1)
    r.raise_for_status()
    return r.json(), elapsed_ms


def render_book_card(book: dict):
    genres_display = ", ".join(book["genres"].split()[:3]) if book.get("genres") else ""
    st.markdown(
        f"""
        <div class="shelf-card">
            <div class="shelf-title">📖 {book['title']}</div>
            <div class="shelf-author">by {book['authors']}</div>
            <div class="shelf-rating">⭐ {book['average_rating']} · {book['ratings_count']:,} ratings</div>
            <div style="font-size:12px;color:#aaa;margin-top:4px;">{genres_display}</div>
            <div class="shelf-reason">💌 {book.get('reason', '')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def book_picker(label: str, key: str):
    query = st.text_input(label, key=f"{key}_query", placeholder="Type a book title...")
    if query and len(query) >= 2:
        matches = search_books(query)
        if matches:
            options = {f"{m['title']} — {m['authors']}": m["book_id"] for m in matches}
            choice = st.selectbox("Pick one:", list(options.keys()), key=f"{key}_choice")
            return options[choice]
    return None


# ===========================================
# Tabs
# ===========================================

tab_similar, tab_foryou, tab_discover, tab_metrics = st.tabs(
    ["💕 Similar Books", "👯 For You", "🌈 Discover", "📊 How It Works"]
)

with tab_similar:
    st.subheader("Loved a book? Find its soulmates 💕")
    st.caption("Content-based matching on genre + author — finds books with a similar vibe.")

    book_id = book_picker("Search for a book you loved:", key="similar")

    if book_id and st.button("Find Similar Books ✨", key="btn_similar"):
        try:
            with st.spinner("Waking up the recommendation engine (may take a minute if idle)..."):
                result, latency_ms = call_similar_books(book_id)

            st.caption(f"⚡ {latency_ms} ms · {'cached' if result.get('cached') else 'freshly computed'}")
            st.markdown(f"**Because you loved:** {result['query']['title']}")
            st.divider()
            for book in result["results"]:
                render_book_card(book)

        except requests.exceptions.RequestException:
            st.error("Could not reach the API — the backend may be waking up. Try again in a moment.")
        except Exception as e:
            st.error(f"Something went wrong: {e}")


with tab_foryou:
    st.subheader("Tell us your faves, we'll find your next obsession 👯")
    st.caption(
        "Real collaborative filtering — based on what actual readers with similar "
        "taste to yours also loved, not just genre matching."
    )

    st.markdown("Pick 2-3 books you loved:")
    liked_ids = []
    for i in range(3):
        bid = book_picker(f"Book {i + 1} (optional after the first)", key=f"foryou_{i}")
        if bid:
            liked_ids.append(bid)

    if liked_ids and st.button("Find My Next Read 🌟", key="btn_foryou"):
        try:
            with st.spinner("Waking up the recommendation engine (may take a minute if idle)..."):
                result, latency_ms = call_for_you(liked_ids)

            st.caption(f"⚡ {latency_ms} ms · {'cached' if result.get('cached') else 'freshly computed'}")
            st.divider()
            if result["results"]:
                for book in result["results"]:
                    render_book_card(book)
            else:
                st.info("Not enough data on these books yet — try different ones!")

        except requests.exceptions.RequestException:
            st.error("Could not reach the API — the backend may be waking up. Try again in a moment.")
        except Exception as e:
            st.error(f"Something went wrong: {e}")


with tab_discover:
    st.subheader("Browse by mood 🌈")
    st.caption(
        "Ranked by a Bayesian-weighted rating (like IMDB uses) — so a book with "
        "5 five-star ratings can't outrank one with 50,000 ratings averaging 4.6."
    )

    genres = st.multiselect("Pick your vibe:", GENRE_OPTIONS, default=["fantasy"])
    min_ratings = st.slider("Minimum ratings count (filters out obscure/unreliable entries):",
                             0, 20000, 5000, step=500)

    if st.button("Discover Books 🔮", key="btn_discover"):
        try:
            with st.spinner("Waking up the recommendation engine (may take a minute if idle)..."):
                result, latency_ms = call_discover(genres, min_ratings)

            st.caption(f"⚡ {latency_ms} ms · {'cached' if result.get('cached') else 'freshly computed'}")
            st.divider()
            if result["results"]:
                for book in result["results"]:
                    render_book_card(book)
            else:
                st.info("No books matched — try loosening your filters!")

        except requests.exceptions.RequestException:
            st.error("Could not reach the API — the backend may be waking up. Try again in a moment.")
        except Exception as e:
            st.error(f"Something went wrong: {e}")


with tab_metrics:
    st.subheader("The real numbers behind ShelfMatch 📊")
    st.caption("No hand-waving — here's how well each approach actually performs, vs. random chance.")

    try:
        r = requests.get(f"{API_URL}/model/metrics", timeout=60)
        r.raise_for_status()
        metrics = r.json()

        st.markdown(f"**Dataset:** {metrics['dataset']}")

        st.markdown("#### 💕 Content-Based Matching")
        cb = metrics["content_based"]
        st.write(f"**Method:** {cb['method']}")
        col1, col2 = st.columns(2)
        col1.metric("Genre overlap@10", f"{cb['model_score'] * 100:.1f}%")
        col2.metric("Random baseline", f"{cb['random_baseline_score'] * 100:.1f}%")

        st.markdown("#### 👯 Collaborative Filtering")
        cf = metrics["collaborative_filtering"]
        st.write(f"**Method:** {cf['method']}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Recall@10", f"{cf['recall_at_10'] * 100:.2f}%")
        col2.metric("Recall@50", f"{cf['recall_at_50'] * 100:.1f}%")
        col3.metric("vs. random", f"{cf['random_baseline_recall_at_10'] * 100:.2f}%")
        st.caption(
            f"Mean percentile rank: {cf['mean_percentile_rank'] * 100:.1f}% "
            f"(0.5% = random guessing — lower is better). "
            f"Evaluated on {cf['n_users_evaluated']} real readers, "
            f"trained on {cf['n_ratings']:,} real ratings from {cf['n_users_total']:,} users."
        )

    except requests.exceptions.RequestException:
        st.error("Could not reach the API to load metrics — the backend may be waking up.")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
