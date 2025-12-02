# recommender/tag_model.py
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_data
def build_tag_based_model(
    movies_df: pd.DataFrame,
    tags_df: pd.DataFrame,
    max_features: int = 5000
):
    tags_df = tags_df.dropna(subset=["tag"]).copy()
    tags_df["tag_clean"] = (
        tags_df["tag"]
        .str.lower()
        .str.strip()
    )

    movie_tags = (
        tags_df
        .groupby("movieId")["tag_clean"]
        .apply(lambda x: " ".join(x))
        .reset_index()
        .rename(columns={"tag_clean": "tags_joined"})
    )

    movies_tags = movies_df.merge(movie_tags, on="movieId", how="left")
    movies_tags["tags_joined"] = movies_tags["tags_joined"].fillna("")

    tfidf = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = tfidf.fit_transform(movies_tags["tags_joined"])

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return movies_tags, cosine_sim


@st.cache_data
def recommend_for_user_tags(
    user_id: int,
    ratings_df: pd.DataFrame,
    movies_tags_df: pd.DataFrame,
    cosine_sim_matrix,
    top_n: int = 10,
    min_rating: float = 4.0
) -> pd.DataFrame:
    user_ratings = ratings_df[ratings_df["userId"] == user_id]
    if user_ratings.empty:
        return pd.DataFrame()

    liked = user_ratings[user_ratings["rating"] >= min_rating]
    if liked.empty:
        return pd.DataFrame()

    movie_id_to_idx = pd.Series(movies_tags_df.index, index=movies_tags_df["movieId"])
    liked_indices = movie_id_to_idx[liked["movieId"]].dropna().astype(int).tolist()
    if not liked_indices:
        return pd.DataFrame()

    sim_scores = np.zeros(cosine_sim_matrix.shape[0])
    for idx in liked_indices:
        sim_scores += cosine_sim_matrix[idx]

    watched_indices = movie_id_to_idx[user_ratings["movieId"]].dropna().astype(int).tolist()
    sim_scores[watched_indices] = -1

    top_indices = np.argsort(sim_scores)[::-1][:top_n]
    scores = sim_scores[top_indices]

    recs = movies_tags_df.iloc[top_indices][["movieId", "title", "genres"]].copy()
    recs["content_score"] = scores
    recs = recs[recs["content_score"] > 0]

    return recs


@st.cache_data
def recommend_for_selected_movies(
    selected_movie_ids,
    movies_tags_df: pd.DataFrame,
    cosine_sim_matrix,
    top_n: int = 10
) -> pd.DataFrame:
    if not selected_movie_ids:
        return pd.DataFrame()

    movie_id_to_idx = pd.Series(movies_tags_df.index, index=movies_tags_df["movieId"])
    liked_indices = movie_id_to_idx[selected_movie_ids].dropna().astype(int).tolist()
    if not liked_indices:
        return pd.DataFrame()

    sim_scores = np.zeros(cosine_sim_matrix.shape[0])
    for idx in liked_indices:
        sim_scores += cosine_sim_matrix[idx]

    sim_scores[liked_indices] = -1

    top_indices = np.argsort(sim_scores)[::-1][:top_n]
    scores = sim_scores[top_indices]

    recs = movies_tags_df.iloc[top_indices][["movieId", "title", "genres"]].copy()
    recs["similarity_score"] = scores
    recs = recs[recs["similarity_score"] > 0]

    return recs
