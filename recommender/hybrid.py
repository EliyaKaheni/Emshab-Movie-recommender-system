# recommender/hybrid.py
import pandas as pd
import numpy as np

from .svdpp_model import recommend_svd_for_user
from .tag_model import recommend_for_user_tags


def hybrid_recommendations(
    user_id: int,
    svd_model,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    movies_tags_df: pd.DataFrame,
    cosine_sim_matrix,
    n: int = 10,
    alpha: float = 0.5,
    min_rating_tag: float = 4.0,
) -> pd.DataFrame:
    """
    α * normalized(SVD++) + (1-α) * normalized(tag-based).
    """
    svd_df = recommend_svd_for_user(
        user_id=user_id,
        model=svd_model,
        movies_df=movies_df,
        n=max(n * 3, 30),
    )
    if svd_df.empty:
        svd_df = pd.DataFrame(columns=["movieId", "pred_rating"])
    svd_df = svd_df.rename(columns={"pred_rating": "svd_score"})

    tag_df = recommend_for_user_tags(
        user_id=user_id,
        ratings_df=ratings_df,
        movies_tags_df=movies_tags_df,
        cosine_sim_matrix=cosine_sim_matrix,
        top_n=max(n * 3, 30),
        min_rating=min_rating_tag,
    )
    if tag_df.empty:
        tag_df = pd.DataFrame(columns=["movieId", "content_score"])

    def normalize(col: pd.Series) -> pd.Series:
        if col.empty:
            return col
        cmin, cmax = col.min(), col.max()
        if cmax == cmin:
            return pd.Series(np.ones(len(col)), index=col.index)
        return (col - cmin) / (cmax - cmin)

    if not svd_df.empty:
        svd_df["svd_norm"] = normalize(svd_df["svd_score"])
    else:
        svd_df["svd_norm"] = []

    if "content_score" in tag_df.columns and not tag_df.empty:
        tag_df["content_norm"] = normalize(tag_df["content_score"])
    else:
        tag_df["content_norm"] = []

    hybrid = pd.merge(
        svd_df[["movieId", "svd_norm"]],
        tag_df[["movieId", "content_norm"]],
        on="movieId",
        how="outer",
    ).fillna(0.0)

    hybrid["hybrid_score"] = alpha * hybrid["svd_norm"] + (1 - alpha) * hybrid["content_norm"]

    hybrid = hybrid.merge(movies_df[["movieId", "title", "genres"]], on="movieId", how="left")
    hybrid = hybrid.sort_values("hybrid_score", ascending=False).head(n)

    return hybrid[["movieId", "title", "genres", "svd_norm", "content_norm", "hybrid_score"]]
