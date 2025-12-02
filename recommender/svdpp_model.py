# recommender/svdpp_model.py
from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import joblib
import streamlit as st


@st.cache_resource
def load_svdpp_model(model_path: str = "models/svd_model.pkl"):
    """
    Load pre-trained Surprise SVD++ model saved with joblib.
    """
    model_file = Path(model_path)
    algo = joblib.load(model_file)
    return algo


def recommend_svd_for_user(
    user_id: int,
    model,
    movies_df: pd.DataFrame,
    n: int = 10
) -> pd.DataFrame:
    """
    Recommend top-n items for an existing user using trained SVD++ model.
    """
    trainset = model.trainset
    raw_uid = user_id

    try:
        inner_uid = trainset.to_inner_uid(raw_uid)
    except ValueError:
        return pd.DataFrame()

    rated_inner_iids = {j for (j, _) in trainset.ur[inner_uid]}

    testset = []
    for inner_iid in trainset.all_items():
        if inner_iid in rated_inner_iids:
            continue
        raw_iid = trainset.to_raw_iid(inner_iid)
        testset.append((raw_uid, raw_iid, 0))

    if not testset:
        return pd.DataFrame()

    predictions = model.test(testset)
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_preds = predictions[:n]

    movie_ids = []
    est_ratings = []
    for p in top_preds:
        try:
            movie_ids.append(int(p.iid))
        except ValueError:
            continue
        est_ratings.append(p.est)

    recs = pd.DataFrame({
        "movieId": movie_ids,
        "pred_rating": est_ratings
    })

    recs = recs.merge(movies_df[["movieId", "title", "genres"]], on="movieId", how="left")
    return recs[["movieId", "title", "genres", "pred_rating"]]
