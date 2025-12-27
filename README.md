# 🎬 Movie Recommender System  
### SVD++, Content-based, Hybrid & Cold-start | NiceGUI Web App (formerly Streamlit)

A complete movie recommender system built on the **MovieLens (ml-latest-small)** dataset.  
It includes **Collaborative Filtering (SVD++)**, **Content-based Filtering (TF-IDF + Cosine Similarity)**, a **Hybrid Recommender**, and a practical **Cold-start solution** for new users — all wrapped in a modern web UI.

---

## ✨ Key Features

### 1) SVD++ Collaborative Filtering
- Trained using the **Surprise** library  
- Learns user/item latent factors and biases  
- Effective on sparse rating matrices  

### 2) Content-based Model (Tags + TF-IDF)
- Aggregates tags per movie (plus title/genres)  
- Builds a TF-IDF representation  
- Uses cosine similarity to recommend similar movies  

### 3) Hybrid Recommender (Scale-robust Rank Fusion)
Instead of mixing raw scores with different scales, the hybrid recommender uses **rank fusion (RRF)** to combine collaborative and content signals in a stable way (no score scale mismatch).

- **α** controls the weight between collaborative vs content  
- Produces robust recommendations even when one model is weaker  

### 4) Cold-start Handling for New Users (Fold-in)
- New users rate a few movies  
- We estimate a user representation using the pretrained SVD++ item factors (fold-in)  
- No full retraining is required for immediate recommendations  
- Optionally blends with content-based recommendations for better cold-start quality  

### 5) Persist New-user Ratings (Optional)
- New-user ratings can be stored (SQLite) and used for periodic retraining  
- Improves personalization over time  

---

## 🏗 Project Structure

```
movie-recommender/
├── app.py
├── requirements.txt
├── README.md
├── scripts/
│   └── retrain.sh
├── models/
│   ├── svd_model.pkl
│   └── reload.flag
├── data/
│   ├── ml-latest-small/
│   └── live_ratings.sqlite
└── recommender/
    ├── __init__.py
    ├── data_loader.py
    ├── svdpp_model.py
    ├── tag_model.py
    ├── hybrid.py
    └── new_user.py
```

---

## 🚀 Quickstart

### 1) (Optional) Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Run the web app
```bash
python app.py
```

The app will be available at:
```text
http://localhost:8080
```

---

## ♻️ Retraining on New Ratings (Scheduled)

If you enable saving new-user ratings to `data/live_ratings.sqlite`, you can periodically retrain the SVD++ model.

### 1) Train / Retrain manually
```bash
python train_svd_model.py
```

### 2) Schedule retraining (cron)
Example: retrain every 6 hours.
```cron
0 */6 * * * /path/to/movie-recommender/scripts/retrain.sh
```

The retraining script can touch `models/reload.flag` so the running app reloads the updated model without a restart.

---

## 📊 Models Overview

### 🟦 SVD++
- Learns latent factors for users & items  
- Predicts ratings for unseen items  

### 🟩 Content-based Filtering
- TF-IDF on (title + genres + aggregated tags)  
- Cosine similarity ranking  

### 🟧 Hybrid Model (RRF)
- Combines rankings instead of raw scores  
- Stable when collaborative and content scores have different scales  
- Final UI score can be mapped to **0–5** for display  

---

## 📦 Dataset: MovieLens (ml-latest-small)

Contains:
- ~100K ratings  
- ~9K movies  
- tags + genres  

---

## ⭐ Support

If you found this project useful, please ⭐ star the repository!
