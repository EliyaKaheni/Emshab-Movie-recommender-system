# app.py
import streamlit as st
import pandas as pd

from recommender.data_loader import load_movielens
from recommender.svdpp_model import load_svdpp_model, recommend_svd_for_user
from recommender.tag_model import (
    build_tag_based_model,
    recommend_for_user_tags,
    recommend_for_selected_movies,
)
from recommender.new_user import recommend_new_user_svdpp
from recommender.hybrid import hybrid_recommendations

# -------------------------
# Global custom CSS
# -------------------------
st.set_page_config(
    page_title="سیستم پیشنهاددهنده فیلم",
    page_icon="🎬",
    layout="wide",
)

def load_css_file(path: str):
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css_file("styles/style.css")


st.title("🎬 به «امشب» خوش اومدین")

# Data
ratings, movies, tags = load_movielens()
movieId_to_title = dict(zip(movies["movieId"], movies["title"]))

# Models
with st.spinner("در حال بارگزاری مدل ..."):
    svdpp_model = load_svdpp_model()

with st.spinner("در حال ساخت مدل ژانر محور ..."):
    movies_tags, cosine_sim = build_tag_based_model(movies, tags)


# Sidebar
st.sidebar.header("⚙️ تنظیمات")
user_ids = sorted(ratings["userId"].unique())
selected_user = st.sidebar.selectbox("انتخاب کاربر:", user_ids)

top_n = st.sidebar.slider("تعداد پیشنهادها: ", 5, 30, 10)

st.sidebar.markdown("---")
alpha = st.sidebar.slider(
    "تنظیم مقدار α",
    0.0,
    1.0,
    0.5,
    0.05,
)

st.sidebar.markdown("---")
min_rating_tag = st.sidebar.slider(
    "کمترین تعداد موارد منتخب ژانر محور:",
    3.0,
    5.0,
    4.0,
    0.5,
)

st.subheader(f"👤کاربر انتخاب شده:{selected_user}")

tab1, tab2, tab3, tab4 = st.tabs(
    ["SVD++ مدل", "مدل ژانر محور", "مدل ترکیبی", "کاربر جدید"]
)

# Tab 1: SVD++
with tab1:
    st.markdown("### 🎯 مدل ++SVD")

    svd_df = recommend_svd_for_user(
        user_id=selected_user,
        model=svdpp_model,
        movies_df=movies,
        n=top_n,
    )

    if svd_df.empty:
        st.info("فیلمی برات پیدا نکردم")
    else:
        st.dataframe(svd_df, use_container_width=True)

# Tab 2: Tag-based
with tab2:
    st.markdown("### 🏷 مدل ژانر محور")

    tag_recs = recommend_for_user_tags(
        user_id=selected_user,
        ratings_df=ratings,
        movies_tags_df=movies_tags,
        cosine_sim_matrix=cosine_sim,
        top_n=top_n,
        min_rating=min_rating_tag,
    )

    if tag_recs.empty:
        st.info("فیلمی برات پیدا نکردم.")
    else:
        st.dataframe(
            tag_recs[["movieId", "title", "genres", "content_score"]],
            use_container_width=True,
        )

# Tab 3: Hybrid
with tab3:
    st.markdown("### ⚗️ مدل ترکیبی")

    hybrid_recs = hybrid_recommendations(
        user_id=selected_user,
        svd_model=svdpp_model,
        ratings_df=ratings,
        movies_df=movies,
        movies_tags_df=movies_tags,
        cosine_sim_matrix=cosine_sim,
        n=top_n,
        alpha=alpha,
        min_rating_tag=min_rating_tag,
    )

    st.dataframe(hybrid_recs, use_container_width=True)

# Tab 4: New User (folding-in SVD++)
with tab4:
    st.markdown("### 👤 کاربر جدید")
    st.markdown(
        "چندتا از فیلم‌هایی که دیدی رو انتخاب کن."
    )

    movie_titles_sorted = movies["title"].sort_values().tolist()
    selected_titles = st.multiselect(
        "انتخاب فیلم‌هایی که دیدی",
        options=movie_titles_sorted,
    )

    new_user_ratings = []

    if selected_titles:
        st.markdown("به فیلمایی که دیدی رای بده")
        for title in selected_titles:
            movie_id = int(movies.loc[movies["title"] == title, "movieId"].iloc[0])
            rating_val = st.slider(
                f"به فیلم «{title}» چه نمره‌ای می‌دی؟",
                min_value=0.5,
                max_value=5.0,
                value=4.0,
                step=0.5,
                key=f"newuser_rating_{movie_id}",
            )
            new_user_ratings.append((movie_id, rating_val))

    if st.button("پیشنهاد فیلم به کاربر جدید"):
        if not new_user_ratings:
            st.info("چه فیلم‌هایی رو قبلا دیدی؟")
        else:
            recs_newuser = recommend_new_user_svdpp(
                new_user_ratings=new_user_ratings,
                model=svdpp_model,
                movies_df=movies,
                top_n=top_n,
                n_epochs=30,
            )

            if recs_newuser.empty:
                st.info("فیلمی برات پیدا نکردم")
            else:
                st.dataframe(recs_newuser, use_container_width=True)

