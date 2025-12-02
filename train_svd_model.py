# train_svdpp.py
import pandas as pd
from surprise import Dataset, Reader, SVDpp
import joblib
from pathlib import Path


DATA_DIR = Path("data/ml-latest-small")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_svdpp_model():
    ratings_path = DATA_DIR / "ratings.csv"
    ratings = pd.read_csv(ratings_path)

    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
    trainset = data.build_full_trainset()

    algo = SVDpp(
        n_factors=50,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02
    )

    print("Training SVD++ model...")
    algo.fit(trainset)

    model_path = MODEL_DIR / "svd_model.pkl"
    joblib.dump(algo, model_path)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    train_svdpp_model()
