from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump as joblib_dump
from joblib import load as joblib_load
from nicegui import app, run, ui
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import sqlite3
import time
import uuid


try:
    from surprise import dump as surprise_dump
except Exception:
    surprise_dump = None


ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("MOVIELENS_DIR", ROOT / "data" / "ml-latest-small"))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", ROOT / "models"))
CACHE_DIR = MODELS_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SVD_MODEL_PATH = MODELS_DIR / "svd_model.pkl"
CONTENT_CACHE_PATH = CACHE_DIR / "content_tfidf.joblib"

LIVE_DB_PATH = Path(os.environ.get("LIVE_DB_PATH", ROOT / "data" / "live_ratings.sqlite"))
LIVE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _db_connect() -> sqlite3.Connection:
    con = sqlite3.connect(LIVE_DB_PATH, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con


def init_live_db() -> None:
    con = _db_connect()
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS live_ratings (
                user_key TEXT NOT NULL,
                movieId  INTEGER NOT NULL,
                rating   REAL NOT NULL,
                ts       INTEGER NOT NULL,
                source   TEXT NOT NULL,
                PRIMARY KEY (user_key, movieId)
            );
            """
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_live_user ON live_ratings(user_key);")
        con.commit()
    finally:
        con.close()


class RecommenderService:
    def __init__(self) -> None:
        self.movies: pd.DataFrame = pd.DataFrame()
        self.ratings: pd.DataFrame = pd.DataFrame()
        self.tags: pd.DataFrame = pd.DataFrame()

        self.movie_ids: np.ndarray = np.array([])
        self.movie_id_to_idx: Dict[int, int] = {}
        self.movie_meta: Dict[int, Dict[str, Any]] = {}

        self.user_ids: List[int] = []
        self.popular_movie_ids: List[int] = []

        self.algo = None

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.X = None

        self._load_data()
        self._load_collab_model()
        self._load_or_build_content_model()

    def _load_data(self) -> None:
        movies_path = DATA_DIR / "movies.csv"
        ratings_path = DATA_DIR / "ratings.csv"
        tags_path = DATA_DIR / "tags.csv"

        if not movies_path.exists() or not ratings_path.exists():
            raise FileNotFoundError(f"MovieLens data not found in: {DATA_DIR}")

        self.movies = pd.read_csv(movies_path)
        self.ratings = pd.read_csv(ratings_path)

        if tags_path.exists():
            self.tags = pd.read_csv(tags_path)
        else:
            self.tags = pd.DataFrame(columns=["userId", "movieId", "tag", "timestamp"])

        self.movies["movieId"] = self.movies["movieId"].astype(int)
        self.ratings["movieId"] = self.ratings["movieId"].astype(int)
        self.ratings["userId"] = self.ratings["userId"].astype(int)

        self.movie_ids = self.movies["movieId"].to_numpy()
        self.movie_id_to_idx = {int(mid): i for i, mid in enumerate(self.movie_ids)}

        for _, r in self.movies.iterrows():
            mid = int(r["movieId"])
            self.movie_meta[mid] = {
                "movieId": mid,
                "title": str(r.get("title", "")),
                "genres": str(r.get("genres", "")).replace("|", " · "),
            }

        self.user_ids = sorted(self.ratings["userId"].unique().tolist())

        counts = self.ratings.groupby("movieId").size().sort_values(ascending=False)
        self.popular_movie_ids = counts.index.astype(int).tolist()

    def _load_collab_model(self) -> None:
        if not SVD_MODEL_PATH.exists():
            self.algo = None
            return

        if surprise_dump is not None:
            try:
                algo = joblib_load(SVD_MODEL_PATH)
                self.algo = algo
                return
            except Exception as e:
                print("\n\n\n\n\nAfsoooss12", e)

                pass

        try:
            with open(SVD_MODEL_PATH, "rb") as f:
                self.algo = joblib_load(f)
        except Exception as e:
            print(e)

            self.algo = None

    def _build_content_text(self) -> pd.Series:
        if len(self.tags) > 0:
            tg = (
                self.tags.dropna(subset=["tag"])
                .assign(tag=lambda d: d["tag"].astype(str))
                .groupby("movieId")["tag"]
                .apply(lambda s: " ".join(s.tolist()))
            )
        else:
            tg = pd.Series(dtype=str)

        movies = self.movies.copy()
        movies["genres"] = movies["genres"].fillna("").astype(str).str.replace("|", " ")
        movies["title"] = movies["title"].fillna("").astype(str)

        movies = movies.merge(tg.rename("tags_text"), how="left", left_on="movieId", right_index=True)
        movies["tags_text"] = movies["tags_text"].fillna("")

        return (movies["title"] + " " + movies["genres"] + " " + movies["tags_text"]).str.strip()

    def _load_or_build_content_model(self) -> None:
        if CONTENT_CACHE_PATH.exists():
            try:
                cached = joblib_load(CONTENT_CACHE_PATH)
                self.vectorizer = cached["vectorizer"]
                self.movie_ids = cached["movie_ids"]
                self.movie_id_to_idx = {int(mid): i for i, mid in enumerate(self.movie_ids)}
                self.X = cached["X"]
                return
            except Exception:
                pass

        content_text = self._build_content_text()

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=80_000,
        )
        X = vectorizer.fit_transform(content_text.tolist())
        X = normalize(X, norm="l2", axis=1)

        self.vectorizer = vectorizer
        self.X = X

        joblib_dump({"vectorizer": self.vectorizer, "movie_ids": self.movie_ids, "X": self.X}, CONTENT_CACHE_PATH)

    def search_movies(self, q: str, limit: int = 30) -> Dict[int, str]:
        q = (q or "").strip().lower()
        if not q:
            mids = self.popular_movie_ids[:limit]
            return {mid: self.movie_meta[mid]["title"] for mid in mids if mid in self.movie_meta}

        m = self.movies.copy()
        m["title_l"] = m["title"].astype(str).str.lower()
        hits = m[m["title_l"].str.contains(q, na=False)].head(limit)
        return {int(r.movieId): str(r.title) for r in hits.itertuples(index=False)}

    def predict_rating(self, user_id: int, movie_id: int) -> Optional[float]:
        if self.algo is None:
            return None

        def ok(pred) -> bool:
            return not getattr(pred, "details", {}).get("was_impossible", False)

        pred = self.algo.predict(user_id, movie_id)
        if not ok(pred):
            pred = self.algo.predict(str(user_id), str(movie_id))
            if not ok(pred):
                return None

        est = float(pred.est)
        return max(0.5, min(5.0, est))


    def recommend_collab(self, user_id: int, n: int = 10, candidate_pool: int = 2500) -> List[Dict[str, Any]]:
        if self.algo is None:
            return []

        rated = set(self.ratings.loc[self.ratings["userId"] == user_id, "movieId"].astype(int).tolist())
        candidates = [mid for mid in self.popular_movie_ids[:candidate_pool] if mid not in rated]
        if len(candidates) == 0:
            candidates = [mid for mid in self.movie_ids.tolist() if mid not in rated][:candidate_pool]

        scored: List[Tuple[int, float]] = []
        for mid in candidates:
            est = self.predict_rating(user_id, int(mid))
            if est is None:
                continue
            scored.append((int(mid), float(est)))

        scored.sort(key=lambda x: x[1], reverse=True)
        out = []
        for mid, est in scored[:n]:
            meta = self.movie_meta.get(mid, {"title": f"movieId={mid}", "genres": ""})
            out.append(
                {
                    "movieId": mid,
                    "title": meta["title"],
                    "genres": meta["genres"],
                    "score": est,
                    "explain": f"پیش‌بینی امتیاز: {est:.2f}",
                }
            )
        return out

    def recommend_content_from_profile(
        self, liked: List[Tuple[int, float]], n: int = 10, exclude_ids: Optional[set] = None
    ) -> List[Dict[str, Any]]:
        if self.X is None or len(liked) == 0:
            return []

        exclude_ids = exclude_ids or set()

        idxs = []
        weights = []
        for mid, r in liked:
            if mid in self.movie_id_to_idx:
                idxs.append(self.movie_id_to_idx[mid])
                weights.append(float(r))
        if len(idxs) == 0:
            return []

        profile = None
        for i, w in zip(idxs, weights):
            v = self.X[i]
            profile = v.multiply(w) if profile is None else profile + v.multiply(w)
        profile = normalize(profile, norm="l2", axis=1)

        sims = profile.dot(self.X.T).toarray().ravel()
        order = np.argsort(-sims)

        out = []
        for j in order:
            mid = int(self.movie_ids[j])
            if mid in exclude_ids:
                continue
            s = float(sims[j])
            if s <= 0:
                break
            meta = self.movie_meta.get(mid, {"title": f"movieId={mid}", "genres": ""})
            out.append(
                {
                    "movieId": mid,
                    "title": meta["title"],
                    "genres": meta["genres"],
                    "score": s,
                    "explain": f"شباهت محتوایی: {s:.3f}",
                }
            )
            if len(out) >= n:
                break
        return out

    def recommend_content_for_user(self, user_id: int, n: int = 10) -> List[Dict[str, Any]]:
        ur = self.ratings.loc[self.ratings["userId"] == user_id, ["movieId", "rating"]]
        if len(ur) == 0:
            return []

        liked = [(int(mid), float(r)) for mid, r in ur.itertuples(index=False) if float(r) >= 3.5]
        if len(liked) == 0:
            liked = [(int(mid), float(r)) for mid, r in ur.itertuples(index=False)]

        exclude_ids = set(ur["movieId"].astype(int).tolist())
        return self.recommend_content_from_profile(liked=liked[:50], n=n, exclude_ids=exclude_ids)

    def recommend_hybrid(self, user_id: int, n: int = 10, alpha: float = 0.6) -> List[Dict[str, Any]]:
        M = max(200, n * 50)
        k = 60

        collab = self.recommend_collab(user_id, n=M) if self.algo is not None else []
        content = self.recommend_content_for_user(user_id, n=M)

        if not collab and not content:
            return []
        if not collab:
            return content[:n]
        if not content:
            return collab[:n]

        rank_c = {int(x["movieId"]): i + 1 for i, x in enumerate(collab)}
        rank_t = {int(x["movieId"]): i + 1 for i, x in enumerate(content)}

        collab_map = {int(x["movieId"]): x for x in collab}
        content_map = {int(x["movieId"]): x for x in content}

        all_ids = set(rank_c.keys()) | set(rank_t.keys())

        fused: List[Tuple[int, float]] = []
        for mid in all_ids:
            s = 0.0
            rc = rank_c.get(mid)
            rt = rank_t.get(mid)
            if rc is not None:
                s += alpha * (1.0 / (k + rc))
            if rt is not None:
                s += (1.0 - alpha) * (1.0 / (k + rt))
            fused.append((mid, s))

        fused.sort(key=lambda x: x[1], reverse=True)

        scores = [s for _, s in fused]
        smin = float(min(scores)) if scores else 0.0
        smax = float(max(scores)) if scores else 0.0

        def to_0_5(s: float) -> float:
            if smax - smin < 1e-12:
                return 2.5
            return (s - smin) / (smax - smin) * 5.0

        out: List[Dict[str, Any]] = []
        for mid, s in fused[:n]:
            meta = self.movie_meta.get(int(mid), {"title": f"movieId={mid}", "genres": ""})
            ce = collab_map.get(mid, {}).get("score")
            ts = content_map.get(mid, {}).get("score")

            expl_parts = []
            if ce is not None:
                expl_parts.append(f"SVD: {float(ce):.2f}")
            if ts is not None:
                expl_parts.append(f"Content: {float(ts):.3f}")
            expl_parts.append(f"RRF: {s:.6f}")

            out.append(
                {
                    "movieId": int(mid),
                    "title": meta["title"],
                    "genres": meta["genres"],
                    "score": float(to_0_5(s)),
                    "explain": " | ".join(expl_parts),
                }
            )

        return out
    
    def _to_inner_iid(self, movie_id: int) -> Optional[int]:
        if self.algo is None:
            return None
        ts = getattr(self.algo, "trainset", None)
        if ts is None:
            return None
        for raw in (movie_id, str(movie_id)):
            try:
                return ts.to_inner_iid(raw)
            except Exception:
                pass
        return None

    def _fold_in_user_mf(self, liked: List[Tuple[int, float]], reg: float = 0.05) -> Optional[Tuple[float, np.ndarray]]:
        if self.algo is None:
            return None
        ts = getattr(self.algo, "trainset", None)
        if ts is None or not hasattr(self.algo, "qi") or not hasattr(self.algo, "bi"):
            return None

        mu = float(ts.global_mean)
        qi = self.algo.qi
        bi = self.algo.bi
        f = qi.shape[1]

        rows = []
        y = []

        for mid, r in liked:
            iid = self._to_inner_iid(int(mid))
            if iid is None:
                continue
            q = qi[iid]
            b_i = float(bi[iid])
            rows.append(np.concatenate(([1.0], q)))
            y.append(float(r) - mu - b_i)

        if len(rows) < 2:
            return None

        A = np.vstack(rows)
        y = np.asarray(y, dtype=float)

        R = np.eye(f + 1, dtype=float)
        R[0, 0] = 0.0
        theta = np.linalg.solve(A.T @ A + reg * R, A.T @ y)

        b_u = float(theta[0])
        p_u = theta[1:].astype(float)
        return b_u, p_u

    def recommend_new_user_collab(
        self,
        liked: List[Tuple[int, float]],
        n: int = 10,
        candidate_pool: int = 2500,
        reg: float = 0.05,
    ) -> List[Dict[str, Any]]:
        if self.algo is None:
            return []

        folded = self._fold_in_user_mf(liked, reg=reg)
        if folded is None:
            return []

        b_u, p_u = folded
        ts = self.algo.trainset
        mu = float(ts.global_mean)
        qi = self.algo.qi
        bi = self.algo.bi

        exclude = {int(mid) for mid, _ in liked}

        candidates = [mid for mid in self.popular_movie_ids[:candidate_pool] if mid not in exclude]
        if not candidates:
            candidates = [int(mid) for mid in self.movie_ids.tolist() if int(mid) not in exclude][:candidate_pool]

        scored: List[Tuple[int, float]] = []
        for mid in candidates:
            iid = self._to_inner_iid(int(mid))
            if iid is None:
                continue
            est = mu + b_u + float(bi[iid]) + float(np.dot(qi[iid], p_u))
            est = max(0.5, min(5.0, float(est)))
            scored.append((int(mid), est))

        scored.sort(key=lambda x: x[1], reverse=True)

        out = []
        for mid, est in scored[:n]:
            meta = self.movie_meta.get(mid, {"title": f"movieId={mid}", "genres": ""})
            out.append(
                {
                    "movieId": mid,
                    "title": meta["title"],
                    "genres": meta["genres"],
                    "score": est,
                    "explain": f"SVD++(fold-in): {est:.2f}",
                }
            )
        return out

    def recommend_new_user_hybrid(
        self,
        liked: List[Tuple[int, float]],
        n: int = 10,
        alpha: float = 0.6,
    ) -> List[Dict[str, Any]]:
        M = max(200, n * 50)
        k = 60

        collab = self.recommend_new_user_collab(liked, n=M) if self.algo is not None else []
        content = self.recommend_content_from_profile(liked=liked, n=M, exclude_ids={int(mid) for mid, _ in liked})

        if not collab and not content:
            return []
        if not collab:
            return content[:n]
        if not content:
            return collab[:n]

        rank_c = {int(x["movieId"]): i + 1 for i, x in enumerate(collab)}
        rank_t = {int(x["movieId"]): i + 1 for i, x in enumerate(content)}

        collab_map = {int(x["movieId"]): x for x in collab}
        content_map = {int(x["movieId"]): x for x in content}

        all_ids = set(rank_c) | set(rank_t)

        fused: List[Tuple[int, float]] = []
        for mid in all_ids:
            s = 0.0
            rc = rank_c.get(mid)
            rt = rank_t.get(mid)
            if rc is not None:
                s += alpha * (1.0 / (k + rc))
            if rt is not None:
                s += (1.0 - alpha) * (1.0 / (k + rt))
            fused.append((mid, s))

        fused.sort(key=lambda x: x[1], reverse=True)

        scores = [s for _, s in fused]
        smin = float(min(scores)) if scores else 0.0
        smax = float(max(scores)) if scores else 0.0

        def to_0_5(s: float) -> float:
            if smax - smin < 1e-12:
                return 2.5
            return (s - smin) / (smax - smin) * 5.0

        out: List[Dict[str, Any]] = []
        for mid, s in fused[:n]:
            meta = self.movie_meta.get(int(mid), {"title": f"movieId={mid}", "genres": ""})
            ce = collab_map.get(mid, {}).get("score")
            ts_ = content_map.get(mid, {}).get("score")

            expl = []
            if ce is not None:
                expl.append(f"SVD: {float(ce):.2f}")
            if ts_ is not None:
                expl.append(f"Content: {float(ts_):.3f}")
            expl.append(f"RRF: {s:.6f}")

            out.append(
                {
                    "movieId": int(mid),
                    "title": meta["title"],
                    "genres": meta["genres"],
                    "score": float(to_0_5(s)),
                    "explain": " | ".join(expl),
                }
            )
        return out
    
    def get_live_ratings(self, user_key: str) -> List[Tuple[int, float]]:
        con = _db_connect()
        try:
            rows = con.execute(
                "SELECT movieId, rating FROM live_ratings WHERE user_key=? ORDER BY ts DESC;",
                (user_key,),
            ).fetchall()
            return [(int(mid), float(r)) for mid, r in rows]
        finally:
            con.close()


    def upsert_live_rating(self, user_key: str, movie_id: int, rating: float, source: str = "onboarding") -> None:
        con = _db_connect()
        try:
            con.execute(
                """
                INSERT INTO live_ratings(user_key, movieId, rating, ts, source)
                VALUES(?,?,?,?,?)
                ON CONFLICT(user_key, movieId) DO UPDATE SET
                    rating=excluded.rating,
                    ts=excluded.ts,
                    source=excluded.source;
                """,
                (user_key, int(movie_id), float(rating), int(time.time()), source),
            )
            con.commit()
        finally:
            con.close()


    def delete_live_rating(self, user_key: str, movie_id: int) -> None:
        con = _db_connect()
        try:
            con.execute("DELETE FROM live_ratings WHERE user_key=? AND movieId=?;", (user_key, int(movie_id)))
            con.commit()
        finally:
            con.close()



def add_global_style() -> None:
    ui.add_head_html(
        """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;600;800&display=swap" rel="stylesheet">
<style>
  body { direction: rtl; font-family: Vazirmatn, ui-sans-serif, system-ui; }
  .kpi { border: 1px solid rgba(0,0,0,0.06); }
  .nice-shadow { box-shadow: 0 10px 30px rgba(0,0,0,0.15); }
</style>
"""
    )


def movie_card(item: Dict[str, Any]) -> None:
    title = item.get("title", "")
    genres = item.get("genres", "")
    explain = item.get("explain", "")
    score = item.get("score", None)

    with ui.card().classes("w-full rounded-2xl kpi nice-shadow"):
        ui.label(title).classes("text-lg font-bold")
        if genres:
            ui.label(genres).classes("text-sm opacity-70")
        ui.separator().classes("my-2")
        with ui.row().classes("items-center justify-between w-full"):
            with ui.badge().classes("px-3 py-1 rounded-full"):
                if isinstance(score, (int, float)):
                    ui.label(f"{score:.3f}" if score <= 1.0 else f"{score:.2f}")
                else:
                    ui.label("-")
            ui.label(explain).classes("text-xs opacity-80")


@dataclass
class NewUserState:
    ratings: List[Tuple[int, float]] = field(default_factory=list)


service: Optional[RecommenderService] = None
service_error: Optional[str] = None


@app.on_startup
def _startup() -> None:
    global service, service_error
    try:
        init_live_db()
        service = RecommenderService()
    except Exception as e:
        service = None
        service_error = str(e)


def build_app(svc: RecommenderService) -> None:
    import uuid

    add_global_style()

    app.storage.user.setdefault("dark_mode", False)
    app.storage.user.setdefault("user_key", uuid.uuid4().hex)
    user_key = str(app.storage.user["user_key"])

    ui.dark_mode(value=bool(app.storage.user["dark_mode"]))
    ui.switch("حالت تیره").bind_value(app.storage.user, "dark_mode").classes("mx-2")

    with ui.header().classes("bg-gradient-to-l from-indigo-600 to-sky-600 text-white"):
        with ui.row().classes("w-full items-center justify-between px-3"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("movie").classes("text-2xl")
                ui.label("سیستم پیشنهاد فیلم «امشب» چی ببینم؟").classes("text-lg font-bold")
            ui.button("GitHub", icon="code", on_click=lambda: ui.open("https://github.com")).props("flat")

    with ui.column().classes("w-full max-w-6xl mx-auto px-4 py-6 gap-4"):
        with ui.row().classes("w-full gap-3"):
            with ui.card().classes("flex-1 rounded-2xl kpi"):
                ui.label("تعداد فیلم‌ها").classes("text-sm opacity-70")
                ui.label(f"{len(svc.movies):,}").classes("text-2xl font-bold")
            with ui.card().classes("flex-1 rounded-2xl kpi"):
                ui.label("تعداد کاربران").classes("text-sm opacity-70")
                ui.label(f"{len(svc.user_ids):,}").classes("text-2xl font-bold")
            with ui.card().classes("flex-1 rounded-2xl kpi"):
                ui.label("Collaborative Model").classes("text-sm opacity-70")
                ui.label("✅ آماده" if svc.algo is not None else "⚠️ پیدا نشد").classes("text-2xl font-bold")

        with ui.tabs().classes("w-full") as tabs:
            tab_existing = ui.tab("کاربر موجود", icon="person")
            tab_new = ui.tab("کاربر جدید", icon="person_add")

        with ui.tab_panels(tabs, value=tab_existing).classes("w-full"):
            with ui.tab_panel(tab_existing):
                with ui.card().classes("w-full rounded-2xl nice-shadow"):
                    ui.label("پیشنهاد برای کاربر موجود").classes("text-xl font-bold")
                    ui.label("UserId را انتخاب کن و روش پیشنهاد را مشخص کن.").classes("text-sm opacity-70")
                    ui.separator().classes("my-3")

                    with ui.row().classes("w-full gap-3 items-end"):
                        user_select = ui.select(
                            options={uid: str(uid) for uid in svc.user_ids[:5000]},
                            label="UserId",
                            value=svc.user_ids[0] if svc.user_ids else None,
                            with_input=True,
                            clearable=False,
                        ).classes("flex-1")

                        method = ui.select(
                            options={
                                "hybrid": "Hybrid",
                                "collab": "Collaborative (SVD++)",
                                "content": "Content-based (Tags/Genres)",
                            },
                            label="روش پیشنهاد",
                            value="hybrid",
                        ).classes("flex-1")

                        topn = ui.number(label="تعداد پیشنهاد", value=12, min=1, max=50, step=1).classes("w-40")

                    alpha_row = ui.row().classes("w-full gap-3 items-center mt-2")
                    with alpha_row:
                        ui.label("وزن Hybrid (α)").classes("text-sm opacity-80")
                        alpha = ui.slider(min=0.0, max=1.0, value=0.6, step=0.05).classes("flex-1")
                        alpha_value = ui.badge("0.60").classes("px-3 py-1 rounded-full")
                        alpha.on("update:model-value", lambda e: alpha_value.set_text(f"{float(e.args):.2f}"))

                    recs_box = ui.element("div").classes("w-full mt-4")
                    progress = ui.linear_progress(show_value=False).classes("w-full")
                    progress.set_visibility(False)

                    async def compute_existing() -> None:
                        recs_box.clear()
                        progress.set_visibility(True)

                        uid = int(user_select.value)
                        m = str(method.value)
                        n = int(topn.value)
                        a = float(alpha.value)

                        def _work() -> List[Dict[str, Any]]:
                            if m == "collab":
                                return svc.recommend_collab(uid, n=n)
                            if m == "content":
                                return svc.recommend_content_for_user(uid, n=n)
                            return svc.recommend_hybrid(uid, n=n, alpha=a)

                        items = await run.io_bound(_work)

                        progress.set_visibility(False)

                        if not items:
                            with recs_box:
                                ui.notify("چیزی برای پیشنهاد پیدا نشد.", type="warning")
                            return

                        with recs_box:
                            ui.label("پیشنهادها").classes("text-lg font-bold mt-2")
                            ui.separator().classes("my-3")
                            with ui.element("div").classes("grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 w-full"):
                                for it in items:
                                    movie_card(it)

                    with ui.row().classes("w-full justify-end mt-3"):
                        ui.button("ساخت پیشنهاد", icon="auto_awesome", on_click=compute_existing).classes(
                            "bg-gradient-to-l from-indigo-600 to-sky-600 text-white rounded-xl"
                        )

            with ui.tab_panel(tab_new):
                state = NewUserState()

                with ui.card().classes("w-full rounded-2xl nice-shadow"):
                    ui.label("پیشنهاد برای کاربر جدید").classes("text-xl font-bold")
                    ui.label("چند فیلم رو انتخاب کن و بهشون امتیاز بده؛ بعد پیشنهاد بگیر.").classes("text-sm opacity-70")
                    ui.separator().classes("my-3")

                    movie_select = ui.select(
                        options=svc.search_movies("", limit=40),
                        label="جستجوی فیلم",
                        with_input=True,
                        clearable=True,
                    ).classes("w-full")

                    async def refresh_options(e) -> None:
                        q = ""
                        if hasattr(e, "args"):
                            if isinstance(e.args, dict) and "value" in e.args:
                                q = str(e.args["value"] or "")
                            else:
                                q = str(e.args or "")
                        movie_select.set_options(svc.search_movies(q, limit=40))

                    movie_select.on("update:model-value", refresh_options)

                    with ui.row().classes("w-full gap-3 items-end mt-2"):
                        rating_in = ui.slider(min=0.5, max=5.0, value=4.0, step=0.5).classes("flex-1")
                        rating_badge = ui.badge("4.0").classes("px-3 py-1 rounded-full")
                        rating_in.on("update:model-value", lambda e: rating_badge.set_text(f"{float(e.args):.1f}"))
                        add_btn = ui.button("اضافه کن", icon="add").classes(
                            "bg-gradient-to-l from-emerald-600 to-teal-600 text-white rounded-xl"
                        )

                    table_cols = [
                        {"name": "title", "label": "فیلم", "field": "title", "align": "left"},
                        {"name": "rating", "label": "امتیاز", "field": "rating", "sortable": True},
                    ]
                    rated_table = ui.table(
                        columns=table_cols, rows=[], row_key="movieId", selection="single", pagination=7
                    ).classes("w-full mt-3")

                    def sync_table() -> None:
                        rows = []
                        for mid, r in state.ratings:
                            meta = svc.movie_meta.get(mid, {"title": f"movieId={mid}"})
                            rows.append({"movieId": mid, "title": meta["title"], "rating": float(r)})
                        rated_table.rows = rows
                        rated_table.update()

                    state.ratings = svc.get_live_ratings(user_key)
                    sync_table()

                    def add_movie() -> None:
                        if movie_select.value is None:
                            ui.notify("اول یک فیلم انتخاب کن.", type="warning")
                            return
                        mid = int(movie_select.value)
                        r = float(rating_in.value)

                        for i, (m0, _) in enumerate(state.ratings):
                            if m0 == mid:
                                state.ratings[i] = (mid, r)
                                svc.upsert_live_rating(user_key, mid, r, source="onboarding")
                                sync_table()
                                ui.notify("امتیاز آپدیت شد ✅")
                                return

                        state.ratings.append((mid, r))
                        svc.upsert_live_rating(user_key, mid, r, source="onboarding")
                        sync_table()
                        ui.notify("اضافه شد ✅")

                    add_btn.on_click(add_movie)

                    def remove_selected() -> None:
                        if not rated_table.selected:
                            ui.notify("یک ردیف رو انتخاب کن.", type="warning")
                            return
                        mid = int(rated_table.selected[0]["movieId"])
                        svc.delete_live_rating(user_key, mid)
                        state.ratings = [(m, r) for (m, r) in state.ratings if m != mid]
                        sync_table()
                        ui.notify("حذف شد 🗑️", type="positive")

                    with ui.row().classes("w-full gap-2 justify-end"):
                        ui.button("حذف انتخاب‌شده", icon="delete", on_click=remove_selected).props("outline")

                    with ui.row().classes("w-full gap-3 items-end mt-4"):
                        n_new = ui.number(label="تعداد پیشنهاد", value=12, min=1, max=50, step=1).classes("w-40")
                        ui.label("پیشنهادها از پروفایل محتوایی ساخته می‌شن.").classes("text-sm opacity-70 flex-1")

                    recs_box2 = ui.element("div").classes("w-full mt-4")
                    progress2 = ui.linear_progress(show_value=False).classes("w-full")
                    progress2.set_visibility(False)

                    async def compute_new_user() -> None:
                        recs_box2.clear()
                        if len(state.ratings) < 2:
                            ui.notify("حداقل ۲ فیلم با امتیاز وارد کن.", type="warning")
                            return

                        progress2.set_visibility(True)
                        n = int(n_new.value)

                        def _work() -> List[Dict[str, Any]]:
                            return svc.recommend_new_user_hybrid(liked=state.ratings, n=n, alpha=0.6)

                        items = await run.io_bound(_work)
                        progress2.set_visibility(False)

                        if not items:
                            ui.notify("چیزی برای پیشنهاد پیدا نشد.", type="warning")
                            return

                        with recs_box2:
                            ui.label("پیشنهادها").classes("text-lg font-bold mt-2")
                            ui.separator().classes("my-3")
                            with ui.element("div").classes("grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 w-full"):
                                for it in items:
                                    movie_card(it)

                    with ui.row().classes("w-full justify-end mt-3"):
                        ui.button("پیشنهاد بده", icon="auto_awesome", on_click=compute_new_user).classes(
                            "bg-gradient-to-l from-indigo-600 to-sky-600 text-white rounded-xl"
                        )


@ui.page("/")
def index() -> None:
    if service is None:
        ui.label("خطا در لود سرویس").classes("text-xl font-bold")
        ui.label(service_error or "unknown error").classes("text-sm opacity-70")
        return
    build_app(service)


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title="Emshab Movie Recommender",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
        storage_secret=os.environ.get("NICEGUI_SECRET", "change-me-in-prod"),
    )
