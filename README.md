# 🎬 Movie Recommender System  
### SVD++, Content-based, Hybrid & Cold-start | Streamlit Web App

This project implements a complete movie recommender system using the MovieLens dataset.  
It includes **Collaborative Filtering (SVD++)**, **Content-based Filtering**, a **Hybrid Model**, and a **Cold-start solution** for new users — all wrapped inside a modern **Streamlit web application**.

---

## 📌 Key Features

### 🔹 1. SVD++ Collaborative Filtering
- Trained using Surprise library  
- Captures implicit and explicit feedback  
- Highly effective for sparse rating datasets  

### 🔹 2. Content-based Model (Tags + TF-IDF)
- Aggregates tags per movie  
- Builds a TF-IDF vector representation  
- Computes cosine similarity between movies  
- Recommends items similar to those the user liked  

### 🔹 3. Hybrid Recommender  
The hybrid model combines both Collaborative Filtering (SVD++) and Content-based similarity to produce more robust recommendations:
\[
\text{HybridScore}(i) = \alpha \cdot \text{SVD++}(i) + (1 - \alpha) \cdot \text{Content}(i)
\]

- **α** controls the weight of each model (configurable in the UI)  
- Provides more **stable and accurate** recommendations than using SVD++ or Content alone  
- Helps balance **behavior-based** and **content-based** signals  


### 🔹 4. New User Cold-start Handling (Folding-in)
- User selects movies they have watched  
- Assigns custom ratings  
- Model updates **only the user’s latent vector (p_u)**  
- No retraining of the main model  
- Fast & practical, used in real-world recommenders  

---

## 🏗 Project Structure

```
movie-recommender/
├── app.py
├── requirements.txt
├── README.md
├── styles/
│   └── style.css
├── models/
│   └── svd_model.pkl
├── data/
│   └── ml-latest-small/
└── recommender/
    ├── __init__.py
    ├── data_loader.py
    ├── svdpp_model.py
    ├── tag_model.py
    ├── hybrid.py
    └── new_user.py
```

---

## 🚀 How to Run

### 1. (Optional) Create a Python virtual environment
```
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Launch the Streamlit app
```
streamlit run app.py
```

App will open automatically at:
```
http://localhost:8501
```

---

## 📊 Models Overview

### 🟦 SVD++
- Learns latent factors for users & items  
- Uses implicit feedback (SVD++)  
- Predicts ratings for unseen items  

### 🟩 Content-based Filtering
- Each movie gets a combined “tag text”  
- TF-IDF vectorization  
- Cosine similarity for recommendation  

### 🟧 Hybrid Model
- Normalizes both SVD++ and Content scores  
- Weighted sum using α  
- More robust and personalized recommendations  

### 🟥 Cold-start Solution (New Users)
- Implements **SVD++ Folding-in**  
- Only updates p_u and b_u  
- Does not require retraining  
- Generates instant recommendations for new users  

---

## 📦 Dataset: MovieLens (ml-latest-small)

Contains:

- ~100,000 ratings  
- ~9,000 movies  
- tags + genres  

---

## ⭐ Support the Project

If you found this project useful, please ⭐ star the repository!

---
