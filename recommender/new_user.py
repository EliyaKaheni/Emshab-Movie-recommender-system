# recommender/new_user.py
import numpy as np
import pandas as pd


def recommend_new_user_svdpp(
    new_user_ratings,        # list[(movieId, rating)]
    model,
    movies_df: pd.DataFrame,
    top_n: int = 10,
    n_epochs: int = 30,
    assumed_min_ratings: int = 1,
) -> pd.DataFrame:
    """
    Folding-in SVD++ for a brand new user (not in trainset).
    """
    trainset = model.trainset
    qi = model.qi
    bi = model.bi
    yj = model.yj
    global_mean = trainset.global_mean

    if len(new_user_ratings) < assumed_min_ratings:
        return pd.DataFrame()

    lr = getattr(model, "lr_pu", 0.005)
    reg = getattr(model, "reg_pu", 0.02)

    n_factors = model.n_factors
    pu = np.zeros(n_factors, dtype=np.float64)
    bu = 0.0

    inner_ratings = []
    inner_items = []

    for movie_id, r_ui in new_user_ratings:
        movie_id = int(movie_id)
        try:
            inner_iid = trainset.to_inner_iid(movie_id)
        except ValueError:
            continue

        inner_ratings.append((inner_iid, float(r_ui)))
        inner_items.append(inner_iid)

    if not inner_ratings:
        return pd.DataFrame()

    inner_items = list(set(inner_items))

    Iu = np.array(inner_items, dtype=np.int64)
    y_sum = np.zeros(n_factors, dtype=np.float64)
    if len(Iu) > 0:
        y_sum = yj[Iu].sum(axis=0) / np.sqrt(len(Iu))

    for _ in range(n_epochs):
        for inner_iid, r_ui in inner_ratings:
            q_i = qi[inner_iid]
            b_i = bi[inner_iid]

            r_hat = global_mean + bu + b_i + np.dot(q_i, pu + y_sum)
            err = r_ui - r_hat

            bu += lr * (err - reg * bu)
            pu += lr * (err * q_i - reg * pu)

    scores = []
    all_inner_items = range(trainset.n_items)
    rated_set = set(inner_items)

    for inner_iid in all_inner_items:
        if inner_iid in rated_set:
            continue
        q_i = qi[inner_iid]
        b_i = bi[inner_iid]
        r_hat = global_mean + bu + b_i + np.dot(q_i, pu + y_sum)
        scores.append((inner_iid, r_hat))

    if not scores:
        return pd.DataFrame()

    scores.sort(key=lambda x: x[1], reverse=True)
    top_items = scores[:top_n]

    raw_movie_ids = []
    est_scores = []
    for inner_iid, score in top_items:
        raw_iid = trainset.to_raw_iid(inner_iid)
        try:
            raw_movie_ids.append(int(raw_iid))
        except ValueError:
            continue
        est_scores.append(score)

    recs = pd.DataFrame({
        "movieId": raw_movie_ids,
        "svdpp_newuser_score": est_scores
    })

    recs = recs.merge(movies_df[["movieId", "title", "genres"]], on="movieId", how="left")
    return recs[["movieId", "title", "genres", "svdpp_newuser_score"]]
