# recommender/data_loader.py
from pathlib import Path
import pandas as pd
import streamlit as st


@st.cache_data
def load_movielens(data_dir: str = "data/ml-latest-small"):
    data_path = Path(data_dir)
    ratings = pd.read_csv(data_path / "ratings.csv")
    movies = pd.read_csv(data_path / "movies.csv")
    tags = pd.read_csv(data_path / "tags.csv")
    return ratings, movies, tags
